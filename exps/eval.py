import cdt, pdb
import jax.numpy as jnp
import numpy as onp
import networkx as nx
import graphical_models
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from scipy.optimize import linear_sum_assignment


def evaluate(rng_key, model_params, LΣ_params, forward, interv_nodes,
            interv_values, z_samples, z_gt, gt_W, gt_sigmas, gt_P, gt_L, 
            loss, log_dict, opt, proj_dims=None, image=False, 
            interv_model=False, generative_interv=False, x_gt=None, gt_interv_targets=None,
            interv_logit_params=None):
    
    eval_mean_args = (
        rng_key, 
        proj_dims, 
        model_params, 
        LΣ_params, 
        forward, 
        interv_nodes, 
        interv_values, 
        z_gt, 
        gt_W, 
        gt_sigmas, 
        gt_P, 
        gt_L, 
        opt, 
    )

    eval_mean_kwargs = {
        "image": image,
        "interv_model": interv_model,
        "generative_interv": generative_interv,
        "x_gt": x_gt,
        "interv_targets": gt_interv_targets,
        "learn_intervs": opt.learn_intervs,
        "interv_logit_params": interv_logit_params
    }

    eval_dict = eval_mean(*eval_mean_args, **eval_mean_kwargs)

    mcc_scores = []
    for j in range(len(z_samples)):
        mcc_scores.append(get_cross_correlation(onp.array(z_samples[j]), onp.array(z_gt)))
    mcc_score = onp.mean(onp.array(mcc_scores))

    wandb_dict = {
                "ELBO": loss,
                "X_MSE": log_dict["x_mse"],
                "KL(L)": log_dict["KL(L)"],
                # "true_obs_KL_term_Z": log_dict["true_obs_KL_term_Z"],
                "train sample KL": eval_dict["sample_kl"],
                
                # Latent representation
                "Z_MSE": log_dict["z_mse"],
                'Evaluations/MCC': mcc_score,

                # Parameter recovery
                "L_MSE": log_dict["L_mse"],

                # Structural metrics
                "Evaluations/SHD": eval_dict["shd"],
                "Evaluations/SHD_C": eval_dict["shd_c"],
                "Evaluations/AUROC": eval_dict["auroc"],
                "Evaluations/AUPRC_W": eval_dict["auprc_w"],
                "Evaluations/AUPRC_G": eval_dict["auprc_g"],

                # Confusion matrix related metrics for structure
                'Evaluations/TPR': eval_dict["tpr"],
                'Evaluations/FPR': eval_dict["fpr"],
                'Evaluations/TP': eval_dict["tp"],
                'Evaluations/FP': eval_dict["fp"],
                'Evaluations/TN': eval_dict["tn"],
                'Evaluations/FN': eval_dict["fn"],
                'Evaluations/Recall': eval_dict["recall"],
                'Evaluations/Precision': eval_dict["precision"],
                'Evaluations/F1 Score': eval_dict["fscore"],
            }

    # Intervention recovery
    if interv_model:
        wandb_dict['Interventions/AUROC'] = eval_dict['Interventions/AUROC']
        wandb_dict['Interventions/AUPRC'] = eval_dict['Interventions/AUPRC']

    return wandb_dict, eval_dict


def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[onp.triu_indices(dim, 1)]
    out_2 = W[onp.tril_indices(dim, -1)]
    return onp.concatenate([out_1, out_2])


def eval_mean(rng_key, proj_dims, model_params, LΣ_params, forward, interv_nodes, 
            interv_values, z_data, gt_W, gt_sigmas, P, L, opt, image=False, 
            interv_model=False, generative_interv=False, x_gt=None, interv_targets=None, 
            learn_intervs=False, interv_logit_params=None):
    """
        Computes mean error statistics for P, L parameters and data
        data should be observational
    """
    edge_threshold = 0.3
    hard = True
    dim = opt.num_nodes
    noise_dim = dim

    Zs = z_data[:opt.obs_data]
    z_prec = onp.linalg.inv(jnp.cov(Zs.T))
    auprcs_w, auprcs_g = [], []
    
    if image:
        forward_args = (model_params, rng_key, hard, proj_dims, rng_key, opt, interv_nodes, interv_values, LΣ_params)
        forward_kwargs = {'P': P}

    elif interv_model:
        if generative_interv:   forward_args = (model_params, rng_key, hard, rng_key, opt, interv_nodes, interv_values, LΣ_params, interv_logit_params)
        else:                   forward_args = (model_params, rng_key, hard, rng_key, opt, x_gt, interv_nodes, interv_values, LΣ_params)
        forward_kwargs = {'P': P, 'L': L, 'learn_intervs': learn_intervs}
        
    else:
        forward_args = (model_params, rng_key, hard, rng_key, opt, interv_nodes, interv_values, LΣ_params)
        forward_kwargs = {'P': P}

    res = forward.apply(*forward_args, **forward_kwargs)

    if image:
        _, _,  _, _, full_l_batch, _, _, W_samples, _ = res
    
    elif interv_model:
        _, _, _, _, full_l_batch, _, _, W_samples, _, extended_interv_target, _, _ = res
        pred_interv_target_samples = extended_interv_target[:, :-1]

    else:
        _, _, _, _, full_l_batch, _, _, W_samples, _ = res

    w_noise = full_l_batch[:, -noise_dim:]

    def sample_stats(est_W, noise, threshold=0.3, get_wasserstein=False):
        est_noise = jnp.ones(dim) * jnp.exp(noise)
        est_W_clipped = jnp.where(jnp.abs(est_W) > threshold, est_W, 0)
        gt_graph_clipped = jnp.where(jnp.abs(gt_W) > threshold, est_W, 0)
        
        binary_est_W = jnp.where(est_W_clipped, 1, 0)
        binary_gt_graph = jnp.where(gt_graph_clipped, 1, 0)
        
        gt_graph_w = nx.from_numpy_array(onp.array(gt_graph_clipped), create_using=nx.DiGraph)
        pred_graph_w = nx.from_numpy_array(onp.array(est_W_clipped), create_using=nx.DiGraph)
        gt_graph_g = nx.from_numpy_array(onp.array(binary_gt_graph), create_using=nx.DiGraph)
        pred_graph_g = nx.from_numpy_array(onp.array(binary_est_W), create_using=nx.DiGraph)
        auprcs_w.append(cdt.metrics.precision_recall(gt_graph_w, pred_graph_w)[0])
        auprcs_g.append(cdt.metrics.precision_recall(gt_graph_g, pred_graph_g)[0])

        stats = count_accuracy(gt_W, est_W_clipped)
        true_KL_divergence = precision_kl_loss(gt_sigmas, gt_W, est_noise, est_W_clipped)
        sample_kl_divergence = precision_kl_sample_loss(z_prec, est_noise, est_W_clipped)

        G_cpdag = graphical_models.DAG.from_nx(nx.DiGraph(onp.array(est_W_clipped))).cpdag()
        gt_graph_cpdag = graphical_models.DAG.from_nx(nx.DiGraph(onp.array(gt_graph_clipped))).cpdag()
        
        stats["shd_c"] = gt_graph_cpdag.shd(G_cpdag)
        stats["sample_kl"] = sample_kl_divergence
        stats["MSE"] = onp.mean((Zs.T - est_W_clipped.T @ Zs.T) ** 2)
        return stats

    stats = sample_stats(W_samples[0], w_noise[0])
    stats = {key: [stats[key]] for key in stats}
    for i, W in enumerate(W_samples[1:]):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(W_samples, gt_W, edge_threshold)
    out_stats["auprc_w"] = onp.array(auprcs_w).mean()
    out_stats["auprc_g"] = onp.array(auprcs_g).mean()

    if interv_model:
        assert len(pred_interv_target_samples.shape) in [2, 3]

        if len(pred_interv_target_samples.shape) == 3:
            pdb.set_trace()
            pred_interv_target_samples = pred_interv_target_samples.mean(axis=0)
        
        precision, recall, _ = precision_recall_curve(interv_targets.ravel(), pred_interv_target_samples.ravel())
        interv_auprc = auc(recall, precision)
        interv_auroc = roc_auc_score(interv_targets.ravel(), pred_interv_target_samples.ravel())
        out_stats['Interventions/AUROC'] = interv_auroc
        out_stats['Interventions/AUPRC'] = interv_auprc

    return out_stats


def gumbel_softmax_logits_auprc(true_interv_targets, interv_gs_logits):
    true_interv_targets = true_interv_targets.reshape(-1)
    interv_gs_logits = interv_gs_logits.reshape(-1)
    interv_auroc = roc_auc_score(true_interv_targets.ravel().ravel())
    return interv_auroc


def auroc(Ws, W_true, threshold=0.3):
    """
        Given a sample of adjacency graphs of shape (n, d, d)
        compute the AUROC for detecting edges. For each edge, we compute
        a probability that there is an edge there which is the frequency with 
        which the sample has edges over threshold.
    """
    _, dim, dim = Ws.shape
    edge_present = jnp.abs(Ws) > threshold
    prob_edge_present = jnp.mean(edge_present, axis=0)
    true_edges = from_W(jnp.abs(W_true) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc_score = auc(fprs, tprs)
    return auroc_score
    

def count_accuracy(W_true, W_est, W_und=None):
    """
    Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Parameters
    ----------
        W_true: (d, d)
            ground truth graph

        W_est: (d, d)
            predicted graph

        W_und
            predicted undirected edges in CPDAG, asymmetric

    Returns as dict
    ---------------
        fdr: 
            False discovery rate = (reverse + false positive) / prediction positive
        tpr: 
            True positive rate = (true positive) / condition positive
        fpr: 
            False positive rate = (reverse + false positive) / condition negative
        shd: 
            Structural Hamming Distance = undirected extra + undirected missing + reverse
        tp:
            True positive
        tn:
            True negative
        fp: 
            False positive
        fn: 
            False negative
        
        Precision
        
        Recall
        
        F1 score:
            2PR / (P + R)
    """
    precision, recall, f1_score = 0., 0., 0.
    B_true = W_true != 0
    B = W_est != 0
    B_und = None if W_und is None else W_und
    d = B.shape[0]
    g_flat = onp.array(B_true.flatten().astype(int))
    pred_flat = onp.array(B.flatten().astype(int))

    # linear index of nonzeros
    pred = onp.flatnonzero(B)
    cond = onp.flatnonzero(B_true)
    cond_reversed = onp.flatnonzero(B_true.T)
    cond_skeleton = onp.concatenate([cond, cond_reversed])
    true_pos = onp.intersect1d(pred, cond, assume_unique=True)
    
    if B_und is not None:
        # treat undirected edge favorably
        pred_und = onp.flatnonzero(B_und)
        true_pos_und = onp.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = onp.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = onp.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = onp.setdiff1d(pred_und, cond_skeleton, assume_unique=True)  # type: ignore
        false_pos = onp.concatenate([false_pos, false_pos_und])
    
    # reverse
    extra = onp.setdiff1d(pred, cond, assume_unique=True)
    reverse = onp.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)  # type: ignore
    
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    # structural hamming distance
    B_lower = onp.tril(B + B.T)
    if B_und is not None:
        B_lower += onp.tril(B_und + B_und.T)
    
    shd = jnp.sum(jnp.abs(g_flat - pred_flat))

    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(g_flat, pred_flat).ravel()
    if int(tp + fp) > 0:    
        precision = tp / (tp + fp)
    if int(tp + fn) > 0:    
        recall = tp / (tp + fn)
    if (precision + recall) > 0.:
        f1_score = 2 * precision * recall / (precision + recall)
    if onp.isnan(f1_score): f1_score = 0
    
    return {"fdr": fdr, 
            "tpr": tpr, 
            "fpr": fpr, 
            "shd": shd, 
            "tp": tp,
            "tn": tn,
            "fp": fp, 
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "fscore": f1_score}


def gaussian_precision_kl(theta, theta_hat):
    """Computes KL divergence between Gaussians with precisions theta, theta_hat

    Assumes mean zero for the Gaussians.

    Args:
        theta: d x d precision matrix
        theta_hat: d x d precision matrix
    Returns:
        kl: KL divergence
    """
    dim = onp.shape(theta)[0]
    theta_0_logdet = onp.linalg.slogdet(theta)[1]
    theta_hat_logdet = onp.linalg.slogdet(theta_hat)[1]
    trace_term = onp.trace(theta_hat @ onp.linalg.inv(theta))
    return 0.5 * (theta_0_logdet - theta_hat_logdet - dim + trace_term)


def precision_kl_loss(true_noise, true_W, est_noise, est_W):
    """Computes the KL divergence to the true parameters

    Args:
        true_noise: (d,)-shape vector of noise variances
        true_W: (d,d)-shape adjacency matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape adjacency matrix
    Returns:
        w_square: induced squared Wasserstein distance
    """
    dim = len(true_noise)
    true_prec = (
        (onp.eye(dim) - true_W) @ (onp.diag(1.0 / true_noise)) @ (onp.eye(dim) - true_W).T
    )
    est_prec = (
        (onp.eye(dim) - est_W) @ (onp.diag(1.0 / est_noise)) @ (onp.eye(dim) - est_W).T
    )
    return gaussian_precision_kl(true_prec, est_prec)


def precision_kl_sample_loss(x_prec, est_noise, est_W):
    """Computes the KL divergence to the sample distribution

    Args:
        x_cov: (d,d)-shape sample covariance matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape estimated adjacency matrix
    Returns:
        w_square: induced KL divergence
    """
    dim = len(est_noise)
    est_prec = (
        (onp.eye(dim) - est_W) @ (onp.diag(1.0 / est_noise)) @ (onp.eye(dim) - est_W).T
    )
    return gaussian_precision_kl(x_prec, est_prec)


def get_cross_correlation(pred_latent, true_latent):
    """
        Calculate the Mean Correlation Coefficient
    """
    dim= pred_latent.shape[1]
    cross_corr= onp.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (onp.cov( pred_latent[:,i], true_latent[:,j] )[0,1]) / ( onp.std(pred_latent[:,i])*onp.std(true_latent[:,j]) )
    
    cost= -1*onp.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    
    score= 100*onp.sum( -1*cost[row_ind, col_ind].sum() )/(dim)
    return score