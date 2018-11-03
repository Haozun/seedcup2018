from config import *
from time import time
import util
import torch as tr
from torch.nn.utils import clip_grad_value_
from torch.nn.functional import cross_entropy as loss_fn


def learning_rate_dacay(optimizer, epoch):
    # learning rate /=2 each two epoch after 10 epochs
    lr = 1e-3 * (0.5 ** max(0, (epoch - 9) // 2))
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def train_model(model, train_iter, val_iter, max_epoch, last_epoch=0):
    optimizer = tr.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # ,lr=0.001
    for epoch_now in range(1 + last_epoch, 1 + max_epoch):
        model.train()
        start_time = time()
        learning_rate_dacay(optimizer, epoch_now)
        step = len(train_iter) // 10
        for i, batch in enumerate(train_iter):
            text = batch.w[0]
            optimizer.zero_grad()
            y_true = [batch.cate1_id, batch.cate2_id, batch.cate3_id]
            y_pred = model(text.to(DEVICE))
            nll_loss_list = [loss_fn(y_pred[i], y_true[i].to(DEVICE)) for i in range(3)]
            tot_loss = wei_criterion(nll_loss_list)
            tot_loss.backward()
            clip_grad_value_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if i % step == 0:
                print(i * BATCH_SIZE, end=' ', flush=True)
                util.log_and_print(nll_loss_list + [tot_loss])

        print("\n %.1f min,turns:%d  " % ((time() - start_time) / 60, epoch_now))

        pred_list = util.get_pred_list(model, val_iter)
        res = util.creterion_val(pred_list)
        util.log_and_print(res)
        tr.save(model.state_dict(), prodirectory + "/{}.pth".format(epoch_now))
