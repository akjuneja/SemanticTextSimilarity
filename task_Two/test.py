from scipy import stats
import logging
import torch
import torch.nn.functional as F



def evaluate_test_set(model, dataloader, config_dict):
    """
    Evaluates the model performance on test data
    """
    # TODO implement
    logging.info("Evaluating accuracy on test set")
    model.load_state_dict(torch.load(config_dict['model_name']))
    running_loss = 0
    total_sentence = 0
    model.eval()
    predicted_scores = []
    true_targets = []
    device = config_dict['device']
    for i, data in enumerate(dataloader['test']):
        sent_a,sent_b,_,_,targets,_,_ = data
        sent_a = sent_a.to(device)
        sent_b = sent_b.to(device)
        targets = targets.to(device)
        similarity_scores, _, _ = model(sent_a, sent_b)
        targets = targets.to('cpu')
        similarity_scores = similarity_scores.to('cpu')
        predicted_scores.extend(similarity_scores.tolist())
        true_targets.extend(targets.tolist())
        
    s = torch.tensor(true_targets, dtype = torch.float32)
    ns = torch.tensor(predicted_scores, dtype = torch.float32, requires_grad = True)
    mse = F.mse_loss(ns, s)
    mes = mse.item()
    sp_cor, _ = stats.spearmanr(predicted_scores,true_targets)
    pr_cor, _ = stats.pearsonr(predicted_scores,true_targets)
    print('Spearman Correlation : ' + str(sp_cor))
    print('Pearson Correlation : ' + str(pr_cor))
    print('Mean Squared Error : ' + str(mse.item()))  
    
