# sys.argv.append("projectName")
from config import *
import util
import train
from model.BNBLSTMr import BNBLSTMr as Model

print("Shop multiClassification::{}".format(prodirectory))
print("Load dataSet")
word_vectors, train_iter, val_iter, test_iter = util.init_workspace()
print("Create Model")
model = Model(embeddings=word_vectors, hidden_dim=HIDDEN_DIM).to(DEVICE)
if LOADMODEL is None:
    model.load_state_dict(tr.load("LOADMODEL"))
print("Start Training")
train.train_model(model, train_iter, val_iter, max_epoch=EPOCHS, last_epoch=LAST_EPOCH)
print("Predict testset")
ans = util.get_pred_list(model, test_iter, use_pandas=True)
ans.to_csv(prodirectory + "/answer.txt", index=False, sep='\t')
print("Answer saved")
