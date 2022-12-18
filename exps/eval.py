import jax.numpy as jnp
import numpy as onp
import networkx as nx
# import cdt
import graphical_models
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import roc_curve, auc
from scipy.optimize import linear_sum_assignment

def evaluate(rng_key, model_params, LΣ_params, forward, interv_nodes,
            interv_values, z_samples, z_gt, gt_W, gt_sigmas, gt_P, loss, log_dict, 
            opt, proj_dims=None, image=False, interv_model=False, x_gt=None):
    
    eval_dict = eval_mean(rng_key, proj_dims, model_params, LΣ_params, forward, 
                            interv_nodes, interv_values, z_gt, gt_W, gt_sigmas, gt_P, 
                            opt, image=image, interv_model=interv_model, x_gt=x_gt)

    mcc_scores = []
    for j in range(len(z_samples)):
        mcc_scores.append(get_cross_correlation(onp.array(z_samples[j]), onp.array(z_gt)))
    mcc_score = onp.mean(onp.array(mcc_scores))

    wandb_dict = {
                "ELBO": loss,
                "Z_MSE": log_dict["z_mse"],
                "X_MSE": log_dict["x_mse"],
                "L_MSE": log_dict["L_mse"],
                "KL(L)": log_dict["KL(L)"],
                "true_obs_KL_term_Z": log_dict["true_obs_KL_term_Z"],
                "train sample KL": eval_dict["sample_kl"],
                "Evaluations/SHD": eval_dict["shd"],
                "Evaluations/SHD_C": eval_dict["shd_c"],
                "Evaluations/AUROC": eval_dict["auroc"],
                "Evaluations/AUPRC_W": eval_dict["auprc_w"],
                "Evaluations/AUPRC_G": eval_dict["auprc_g"],
                'Evaluations/MCC': mcc_score,

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
    
    return wandb_dict, eval_dict

def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[onp.triu_indices(dim, 1)]
    out_2 = W[onp.tril_indices(dim, -1)]
    return onp.concatenate([out_1, out_2])


def eval_mean(rng_key, proj_dims, model_params, LΣ_params, forward, interv_nodes, 
            interv_values, z_data, gt_W, gt_sigmas, P, opt, image=False, interv_model=False, x_gt=None):
    """
        Computes mean error statistics for P, L parameters and data
        data should be observational
    """
    edge_threshold = 0.3
    hard = True
    dim = opt.num_nodes
    if opt.eq_noise_var: noise_dim = 1
    else: noise_dim = dim

    z_data = z_data[:opt.obs_data]
    z_prec = onp.linalg.inv(jnp.cov(z_data.T))
    Zs = z_data[:opt.obs_data]
    auprcs_w, auprcs_g = [], []

    if image:
        (   
            pred_X,
            _, 
            _,
            batched_qz_samples,
            full_l_batch, 
            full_log_prob_l,
            L_samples, 
            W_samples, 
            log_noise_std_samples
                                    ) = forward.apply(model_params, 
                                                        rng_key, 
                                                        hard, 
                                                        proj_dims, 
                                                        rng_key, 
                                                        opt, 
                                                        interv_nodes, 
                                                        interv_values, 
                                                        LΣ_params, 
                                                        P=P)
    
    elif interv_model:
        (   
            pred_X, _, _, batched_qz_samples, full_l_batch, 
            full_log_prob_l, L_samples, W_samples, log_noise_std_samples, _
                                                                            ) = forward.apply(model_params, 
                                                                                                rng_key,
                                                                                                hard,
                                                                                                rng_key,
                                                                                                opt,
                                                                                                x_gt,
                                                                                                interv_values, 
                                                                                                LΣ_params, 
                                                                                                P=P)


    else: 
        (   
            pred_X,
            _, 
            _,
            batched_qz_samples,
            full_l_batch, 
            full_log_prob_l,
            L_samples, 
            W_samples, 
            log_noise_std_samples
                                    ) = forward.apply(model_params, 
                                                        rng_key, 
                                                        hard, 
                                                        rng_key, 
                                                        opt, 
                                                        interv_nodes, 
                                                        interv_values, 
                                                        LΣ_params, 
                                                        P=P)

    w_noise = full_l_batch[:, -noise_dim:]

    def sample_stats(est_W, noise, threshold=0.3, get_wasserstein=False):
        est_noise = jnp.ones(dim) * jnp.exp(noise)
        est_W_clipped = jnp.where(jnp.abs(est_W) > threshold, est_W, 0)
        gt_graph_clipped = jnp.where(jnp.abs(gt_W) > threshold, est_W, 0)
        
        binary_est_W = jnp.where(est_W_clipped, 1, 0)
        binary_gt_graph = jnp.where(gt_graph_clipped, 1, 0)

        gt_graph_w = nx.from_numpy_matrix(onp.array(gt_graph_clipped), create_using=nx.DiGraph)
        pred_graph_w = nx.from_numpy_matrix(onp.array(est_W_clipped), create_using=nx.DiGraph)

        gt_graph_g = nx.from_numpy_matrix(onp.array(binary_gt_graph), create_using=nx.DiGraph)
        pred_graph_g = nx.from_numpy_matrix(onp.array(binary_est_W), create_using=nx.DiGraph)

        # auprcs_w.append(cdt.metrics.precision_recall(gt_graph_w, pred_graph_w)[0])
        # auprcs_g.append(cdt.metrics.precision_recall(gt_graph_g, pred_graph_g)[0])

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
    # out_stats["auprc_w"] = onp.array(auprcs_w).mean()
    # out_stats["auprc_g"] = onp.array(auprcs_g).mean()
    return out_stats


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
    
    try:
        pred_lower = onp.flatnonzero(B_lower)
        cond_lower = onp.flatnonzero(onp.tril(B_true + B_true.T))
        extra_lower = onp.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = onp.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
    except:
        shd = onp.sum(onp.abs(g_flat - pred_flat))
        
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