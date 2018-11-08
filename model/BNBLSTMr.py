from config import *
import torch as tr
import torch.nn as nn

class BNBLSTMr(nn.Module):
    def __init__(self, embeddings, hidden_dim):
        super(BNBLSTMr, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = BATCH_SIZE
        self.word_embeddings = nn.Embedding.from_pretrained(tr.FloatTensor(embeddings))
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT
        self.bilstm = nn.LSTM(EMBEDDING_DIM, self.hidden_dim // 2,
                              num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.class1 = nn.Linear(self.hidden_dim, CATE1_NUM)
        self.class2 = nn.Linear(self.hidden_dim, CATE2_NUM)
        self.class3 = nn.Linear(self.hidden_dim, CATE3_NUM)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

    def attention(self, rnn_out, h_n):
        state = h_n[-2:]  # 2x64x128 select last layer
        merged_state = tr.cat([s for s in state], 1)  # 64x256 merge two direction
        rnn_out_weights = tr.bmm(rnn_out, merged_state.unsqueeze(2)).squeeze(2)  # 64x200
        rnn_out_weights = nn.functional.softmax(rnn_out_weights, 1)
        new_hidden_state = tr.bmm(rnn_out.transpose(1, 2), rnn_out_weights.unsqueeze(2)).squeeze(2)  # 64x256
        return new_hidden_state

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        h0 = tr.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = tr.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).to(DEVICE)
        return h0, c0

    def forward(self, input_sentences):  # input: 64x200
        embeds = self.word_embeddings(input_sentences)  # 64x200x300
        x = embeds.permute(1, 0, 2)  # 200x64x300
        output, (h_n, c_n) = self.bilstm(x, self.init_hidden(x.shape[1]))  # 200x64x256 + (4x64x128 + 4x64x128)
        output = output.permute(1, 0, 2)  # 64x200x256
        final_out = self.attention(output, h_n)  # 64x256
        final_out = self.fc(final_out)
        pred1 = self.class1(final_out)
        pred2 = self.class2(final_out)
        pred3 = self.class3(final_out)
        return pred1, pred2, pred3
