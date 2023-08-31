import torch 
from utils.utils import * 
import tqdm


def evaluate(dataloader,model,device='cpu',conf_threshold=0.001,iou_threshold=0.6):
        model.eval().to(device)
        labels = []
        sample_metrics = []

        with torch.no_grad(), tqdm.tqdm(total=len(dataloader), desc="Evaluating") as pbar:
                for j, (x, y) in enumerate(dataloader):
                        targets=process_targets(y)
                        labels+=targets[:,-1]


                        x = x.to(device)
                        preds = model(x)
                        preds = preds.detach().cpu()
                        output=non_max_suppression(preds,conf_threshold,iou_threshold)
                        sample_metrics+=get_batch_statistics(output,targets,iou_threshold)
                        pbar.update(1)  # Update progress bar

                true_positives, pred_scores, pred_labels = [
                        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
                metrics_output = ap_per_class(
                        true_positives, pred_scores, pred_labels, labels)
                print_eval_stats(metrics_output, class_names, True)

                return metrics_output
        
