import os
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import time
import cv2
from mapillary_sls_main.evaluate import eval_api, create_dummy_predictions


def test_efficient_ram_usage(args, eval_ds, model, test_method="hard_resize"):
    """This function gives the same output as test(), but uses much less RAM.
    This can be useful when testing with large descriptors (e.g. NetVLAD) on large datasets (e.g. San Francisco).
    Obviously it is slower than test(), and can't be used with PCA.
    """
    
    model = model.eval()
    if test_method == 'nearest_crop' or test_method == "maj_voting":
        distances = np.empty([eval_ds.queries_num * 5, eval_ds.database_num], dtype=np.float32)
    else:
        distances = np.empty([eval_ds.queries_num, eval_ds.database_num], dtype=np.float32)

    with torch.no_grad():
        if test_method == 'nearest_crop' or test_method == 'maj_voting':
            queries_features = np.ones((eval_ds.queries_num * 5, args.features_dim), dtype="float32")
        else:
            queries_features = np.ones((eval_ds.queries_num, args.features_dim), dtype="float32")
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            if test_method == "nearest_crop" or test_method == 'maj_voting':
                start_idx = (indices[0] - eval_ds.database_num) * 5
                end_idx = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                queries_features[indices, :] = features.cpu().numpy()
            else:
                queries_features[indices.numpy()-eval_ds.database_num, :] = features.cpu().numpy()

        queries_features = torch.tensor(queries_features).type(torch.float32).cuda()
        
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            inputs = inputs.to(args.device)
            features = model(inputs)
            for pn, (index, pred_feature) in enumerate(zip(indices, features)):
                distances[:, index] = ((queries_features-pred_feature)**2).sum(1).cpu().numpy()
        del features, queries_features, pred_feature
        
    predictions = distances.argsort(axis=1)[:, :max(args.recall_values)]
    
    if test_method == 'nearest_crop':
        distances = np.array([distances[row, index] for row, index in enumerate(predictions)])
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each
    elif test_method == 'maj_voting':
        distances = np.array([distances[row, index] for row, index in enumerate(predictions)])
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query
    del distances
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        extraction_time = 0
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            t_s = time.time()
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
            extraction_time += (time.time() - t_s)
        print('extraction time for each', extraction_time/eval_ds.database_num, extraction_time, eval_ds.database_num)
        logging.debug("Extracting queries features for evaluation/testing, extraction time for each{}".format(extraction_time/eval_ds.database_num))
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx   = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    print('\n All feature', all_features.shape, all_features.dtype, all_features.nbytes)
    print('Memory:', all_features.nbytes)
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    t_s = time.time()
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    logging.debug("Retrieval time for each:{}".format((time.time() - t_s) / queries_features.shape[0]))
    print("Retrieval time for each:{}".format((time.time() - t_s) / queries_features.shape[0]))
    
    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    if args.dataset_name == 'msls':
        predictions_str = []
        for query_index, pred in enumerate(predictions):
            string = [eval_ds.queries_paths[query_index].split('/')[-1].replace('.jpg','')]
            for pred_id in pred:
                string.append(eval_ds.database_paths[pred_id].split('/')[-1].replace('.jpg',''))
            predictions_str.append(string)
            # if query_index ==0:
            #     print(string, predictions_str)
            # raise Exception
        predictions_str = np.array(predictions_str)
        if eval_ds.split == 'test':
            create_dummy_predictions(prediction_path=os.path.join(args.save_dir, 'global.csv'), dataset=eval_ds, ranks=predictions_str[:,1:200])
            return [0], ''
        else:
            recalls_str, recalls = eval_api(predictions_str, ks=args.recall_values, root_default=eval_ds.dataset_folder) #eval_ds.dataset_folder
            return recalls, recalls_str

    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n


def test_rerank(args, eval_ds, model, test_method="hard_resize", pca=None, num_local=500, rerank_dim=131, rerank_top=100, rerank_bs=4, save=None, reg_top=5,ransac=False, threshold=0, debug=False):
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                           "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"

    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim),
                                    dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        all_features_rerank = np.empty((len(eval_ds), num_local, rerank_dim), dtype="float32") # 33 , 65
        extraction_time = 0
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            t_s = time.time()
            features, re_features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            re_features = re_features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
            all_features_rerank[indices.numpy(), :, :] = re_features
            extraction_time += (time.time() - t_s)
        logging.debug("Extracting queries features for evaluation/testing, time for each:{}".format(extraction_time/eval_ds.database_num))
        print('extraction time:', extraction_time/eval_ds.database_num, extraction_time, eval_ds.database_num)
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features, re_features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            re_features = re_features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)

            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
                all_features_rerank[indices.numpy(), :, :] = re_features

    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    queries_re_features = all_features_rerank[eval_ds.database_num:]
    database_re_features = all_features_rerank[:eval_ds.database_num]
    print('\n All feature',all_features_rerank.shape, all_features_rerank.dtype, all_features_rerank.nbytes)
    print('Memory:', all_features.nbytes+all_features_rerank.nbytes)
    if save is not None:
        np.save(save+'_queries_features.npy', queries_features)
        np.save(save+'_database_features.npy', database_features)
        np.save(save+'_queries_re_features.npy', queries_re_features)
        np.save(save+'_database_re_features.npy', database_re_features)

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del all_features, all_features_rerank

    logging.debug("Calculating recalls")
    t_s = time.time()
    distances, predictions = faiss_index.search(queries_features, rerank_top)  # max(args.recall_values)
    retrieval_time = (time.time() - t_s)
    print("Retrieval time for each:{}".format(retrieval_time / queries_features.shape[0]), retrieval_time,
          queries_features.shape[0], args.features_dim)
    logging.debug("Retrieval time for each:{},{}/{}".format(retrieval_time/queries_features.shape[0], retrieval_time,
                  queries_features.shape[0]))

    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    if args.dataset_name == 'msls':
        predictions_str = []
        for query_index, pred in enumerate(predictions):
            string = [eval_ds.queries_paths[query_index].split('/')[-1].replace('.jpg','')]
            for pred_id in pred:
                string.append(eval_ds.database_paths[pred_id].split('/')[-1].replace('.jpg',''))
            predictions_str.append(string)
        predictions_str = np.array(predictions_str)
        if eval_ds.split == 'test':
            create_dummy_predictions(prediction_path=os.path.join(args.save_dir, 'global.csv'), dataset=eval_ds, ranks=predictions_str[:,1:200])
            # return [0], ''
        else:
            recalls_str, recalls = eval_api(predictions_str, ks=args.recall_values, root_default=eval_ds.dataset_folder)
    else:
        recalls = np.zeros(len(args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
        # Divide by the number of queries*100, so the recalls are in percentages
        recalls = recalls / eval_ds.queries_num * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    if save is not None:
        return predictions, recalls_str
    # ====================================================================
    # new rerank
    Reranker = torch.nn.DataParallel(model.module.Reranker)
    Reranker.eval()
    sm = torch.nn.Softmax(dim=1)
    similarity = 1. - distances/2.
    ranks = np.array(predictions).copy()
    new_rank = np.copy(ranks)
    rerank_time = 0
    with torch.no_grad():
        for query_index in tqdm(range(0, predictions.shape[0], rerank_bs), ncols=100):
            # print(query_index)
            query_inputs = queries_features[query_index:min(predictions.shape[0], query_index + rerank_bs)]
            query_inputs_expand = np.tile(np.expand_dims(query_inputs, 1), [1, rerank_top, 1]).reshape(
                [-1, queries_features.shape[-1]])
            query_re_inputs = queries_re_features[query_index:min(predictions.shape[0], query_index + rerank_bs)]
            query_re_inputs_expand = np.tile(np.expand_dims(query_re_inputs, 1), [1, rerank_top, 1, 1]).reshape(
                [-1, queries_re_features.shape[-2], queries_re_features.shape[-1]])

            candidate_index = predictions[query_index:min(predictions.shape[0], query_index + rerank_bs), :rerank_top]
            candidate_re_inputs = database_re_features[candidate_index.reshape(-1)]
            candidate_inputs = database_features[candidate_index.reshape(-1)]
            # =============================================================================
            query_inputs_cuda = torch.tensor(query_inputs_expand).cuda()
            query_re_inputs_cuda = torch.tensor(query_re_inputs_expand.astype(np.float32)).cuda()
            candidate_inputs_cuda = torch.tensor(candidate_inputs).cuda()
            candidate_re_inputs_cuda = torch.tensor(candidate_re_inputs.astype(np.float32)).cuda()
            # =============================================================================
            time_s = time.time()
            rerank_score_ori, final_score = Reranker(query_inputs_cuda, query_re_inputs_cuda,
                                    candidate_inputs_cuda, candidate_re_inputs_cuda)
            rerank_score = sm(rerank_score_ori)[:, 1]
            rerank_score = torch.reshape(rerank_score,[query_re_inputs.shape[0], rerank_top]).detach()#.cpu().numpy()  # .softmax(dim=1)

            for id, candidates in enumerate(candidate_index):
                global_score = torch.tensor(similarity[query_index + id]).cuda()
                rerank_order = torch.argsort(-rerank_score[id]).cpu().numpy()
                new_rank[query_index + id, :rerank_top] = ranks[query_index + id, :rerank_top][rerank_order]
            rerank_time += (time.time() - time_s)

    logging.debug(
        f"{new_rank.shape}, {rerank_score.shape}, time:,{time.time() - time_s}, time each:, {rerank_time / predictions.shape[0]}")
    print(f"time:,{rerank_time}, time each:, {rerank_time / predictions.shape[0]}")
    #==========================================================================================================

    if args.dataset_name == 'msls':
        predictions_str = []
        for query_index, pred in enumerate(new_rank):
            string = [eval_ds.queries_paths[query_index].split('/')[-1].replace('.jpg', '')]
            for pred_id in pred:
                string.append(eval_ds.database_paths[pred_id].split('/')[-1].replace('.jpg', ''))
            predictions_str.append(string)
        predictions_str = np.array(predictions_str)
        if eval_ds.split == 'test':
            create_dummy_predictions(prediction_path=os.path.join(args.save_dir, 'rerank.csv'), dataset=eval_ds, ranks=predictions_str[:,1:200])
            return [0], ''
        else:
            recalls_str, recalls = eval_api(np.array(predictions_str), ks=args.recall_values, root_default=eval_ds.dataset_folder) #eval_ds.dataset_folder

    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(new_rank):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = recalls_str + '\n' + 'rerank: ' + ", ".join(
        [f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    recalls_retrieval = recalls
    # =========================================================================================
    if debug:
        return ranks, new_rank
    return recalls_retrieval, recalls_str


def compare_two_ransac(qfeats, dbfeats, qkeypoints, dbkeypoints):
    scores = []
    all_inlier_index_keypoints = []
    all_inlier_query_keypoints = []
    # for qfeat, dbfeat, qkeypoint, dbkeypoint in zip(qfeats, dbfeats, qkeypoints, dbkeypoints):
    fw_inds, bw_inds = dist_x_y(qfeats, dbfeats) # torch_nn(qfeat, dbfeat)

    # fw_inds = fw_inds.cpu().numpy()
    # bw_inds = bw_inds.cpu().numpy()

    mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

    if len(mutuals) > 3: # need at least four points to estimate a Homography
        index_keypoints = dbkeypoints[mutuals]
        query_keypoints = qkeypoints[fw_inds[mutuals]]

        # index_keypoints = np.transpose(index_keypoints)
        # query_keypoints = np.transpose(query_keypoints)
        # print(index_keypoints, query_keypoints, index_keypoints.shape, query_keypoints.shape)
        _, mask = cv2.findHomography(index_keypoints, query_keypoints, cv2.FM_RANSAC,
                                     ransacReprojThreshold=16*1.5)

        # RANSAC reproj threshold is set to the (stride*1.5) in image space for vgg-16, given a particular patch stride
        # in this work, we ignore the H matrix output - but users of this code are welcome to utilise this for
        # pose estimation (something we may also investigate in future work)

        inlier_index_keypoints = index_keypoints[mask.ravel() == 1]
        all_inlier_query_keypoints.append(query_keypoints[mask.ravel() == 1])
        inlier_count = inlier_index_keypoints.shape[0]
        scores.append(inlier_count / qfeats.shape[0])
        all_inlier_index_keypoints.append(inlier_index_keypoints)
        # we flip to negative such that best match is the smallest number, to be consistent with vanilla NetVlad
        # we normalise by patch count to remove biases in the scoring between different patch sizes (so that all
        # patch sizes are weighted equally and that the only weighting is from the user-defined patch weights)
        # else:
        #     scores.append(0.)
    # return scores[0], index_keypoints, query_keypoints
    return scores[0], all_inlier_query_keypoints[0], all_inlier_index_keypoints[0]