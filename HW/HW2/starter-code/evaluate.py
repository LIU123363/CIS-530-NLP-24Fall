import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from argparse import ArgumentParser


def warn(*args, **kwargs): pass
import warnings; warnings.warn = warn


parser = ArgumentParser()

parser.add_argument("-p", "--predicted", dest = "pred_path",
    required = True, help = "path to your model's predicted labels file")

parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")

parser.add_argument("-c", "--confusion", dest = "show_confusion",
    action = "store_true", help = "show confusion matrix")

args = parser.parse_args()


pred = pd.read_csv(args.pred_path, index_col = "id")
dev  = pd.read_csv(args.dev_path,  index_col = "id")

pred.columns = ["predicted"]
dev.columns  = ["actual"]

data = dev.join(pred)


if args.show_confusion:
    
    data["count"] = 1
    counts = data.groupby(["actual", "predicted"]).count().reset_index()
    confusion = counts[counts.actual != counts.predicted].reset_index(drop = True)
        
    print("Confusion Matrix:")
    
    if confusion.empty: print("None!")
    else: print(confusion)

else:

    print("Mean F1 Score:", f1_score(
        data.actual,
        data.predicted,
        average = "weighted"
    ))
