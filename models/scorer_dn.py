import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic import masked_softmax, to_tensor

from models.graph_transformer import GraphTransformer
from models.models import (GAT, GEAT, Attention, CQAttention,
                           PretrainedEmbeddings, RelationEncoder,
                           SelfAttention, TokenEncoder)


class CommandScorerWithDN(nn.Module):
    def __init__(self, word_emb, hidden_size, device, dropout_ratio=None, word2id=None, value_network='local', diff_network='diffg', diff_no_elu=False):
        super(CommandScorerWithDN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.word2id = word2id
        self.word_emb = word_emb
        self.rnn_hidden_size = int(hidden_size / 2)
        self.value_network = value_network
        self.diff_network = diff_network
        self.activation = nn.ELU
        self.diff_no_elu = diff_no_elu

        if dropout_ratio is None:
            self.dropout_ratio = 0.0 # *
        else:
            self.dropout_ratio = dropout_ratio
        self.n_heads = 1 # *
        self.bidirectional = True
        bi_factor = (2 if self.bidirectional else 1) # hidden size multiplier when bidirectional is used

        self.word_embedding = PretrainedEmbeddings(word_emb)
        self.word_embedding_size = self.word_embedding.dim  # *
        self.word_embedding_prj = torch.nn.Linear(self.word_embedding_size, self.hidden_size, bias=False)
        
        self.dn_gru = nn.GRU(hidden_size, self.rnn_hidden_size, batch_first=True, bidirectional= self.bidirectional)
        # self.cmd_encoder_gru = nn.GRU(hidden_size, self.rnn_hidden_size, batch_first=True, bidirectional= self.bidirectional)
        
        self.param_size = self.hidden_size
        self.W_e = nn.Parameter(torch.zeros(size=(self.param_size, self.param_size)))
        nn.init.xavier_uniform_(self.W_e.data, gain=1.414)
        self.W_l = nn.Parameter(torch.zeros(size=(self.param_size, self.param_size)))
        nn.init.xavier_uniform_(self.W_e.data, gain=1.414)
        self.W_w = nn.Parameter(torch.zeros(size=(self.param_size, self.param_size)))
        nn.init.xavier_uniform_(self.W_e.data, gain=1.414)
        
        self.graph_prj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                       self.activation(),
                                       nn.Dropout(p=0.1))
        
        if self.value_network == 'obs':
            # Encoder for th observation
            self.encoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional= self.bidirectional)
            self.state_gru = nn.GRU(hidden_size*bi_factor, hidden_size*bi_factor, batch_first=True)
            self.state_hidden = []
        elif self.value_network == 'diff':
            self.diff_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional= self.bidirectional)
        elif self.value_network == 'local':
            self.localkg_gat = GAT(hidden_size, hidden_size, self.dropout_ratio, alpha=0.2, nheads=self.n_heads)
            self.local_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional= self.bidirectional)
        self.critic = nn.Linear(hidden_size*bi_factor, 1)

        self.general_attention = Attention(hidden_size, hidden_size)

        if dropout_ratio is None:
            self.att_cmd = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size,1))
        else:
            self.att_cmd = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size * 2),
                                         self.activation(),
                                         nn.Dropout(p=self.dropout_ratio),
                                         nn.Linear(hidden_size * 2, 1))

    def forward(self, obs, commands, local_graph, local_adj, entities, locals, worlds, **kwargs):
        batch_size = commands.size(0)
        nb_cmds = commands.size(1)
        nb_items = entities.size(1)
        
        entities_embedded = self.word_embedding(entities)
        entities_embedded_proj = self.word_embedding_prj(entities_embedded)
        locals_embedded = self.word_embedding(locals)
        locals_embedded_proj = self.word_embedding_prj(locals_embedded)
        graph_encoding_list = []
        for b in range(batch_size):
            entities_output, entities_hidden = self.dn_gru(entities_embedded_proj[b])
            entities_hidden = entities_hidden.permute(1, 0, 2).reshape(entities_hidden.shape[1], -1) if entities_hidden.shape[0] == 2 else entities_hidden
            locals_output, locals_hidden = self.dn_gru(locals_embedded_proj[b])
            locals_hidden = locals_hidden.permute(1, 0, 2).reshape(locals_hidden.shape[1], -1) if locals_hidden.shape[0] == 2 else locals_hidden
            worlds_hidden_list = []
            for e in range(nb_items):
                worlds_embedded = self.word_embedding(worlds[b][e])
                worlds_embedded_proj = self.word_embedding_prj(worlds_embedded)
                worlds_output, worlds_hidden = self.dn_gru(worlds_embedded_proj)
                worlds_hidden = worlds_hidden.permute(1, 0, 2).reshape(worlds_hidden.shape[1], -1) if worlds_hidden.shape[0] == 2 else worlds_hidden
                worlds_hidden_list.append(torch.sum(torch.matmul(worlds_hidden, self.W_w), dim=0))
            if self.diff_no_elu:
                graph_hidden = torch.matmul(entities_hidden, (torch.ones(size=(self.param_size, self.param_size), device=self.device) + self.W_e)) + torch.matmul(locals_hidden, self.W_l) + torch.stack(worlds_hidden_list, dim=0)
                graph_encoding_list.append(graph_hidden)
            else:
                graph_hidden = F.elu(torch.matmul(entities_hidden, (torch.ones(size=(self.param_size, self.param_size), device=self.device) + self.W_e))) + F.elu(torch.matmul(locals_hidden, self.W_l)) + torch.stack(worlds_hidden_list, dim=0)
                graph_encoding_list.append(graph_hidden)
        if self.diff_no_elu:
            graph_encoding_ = torch.stack(graph_encoding_list, dim=0)
        else:
            graph_encoding_ = F.elu(torch.stack(graph_encoding_list, dim=0))
        graph_encoding = self.graph_prj(graph_encoding_)
                
        if self.value_network == 'obs':
            # Observed State
            embedded = self.word_embedding(obs) # batch x word x emb_size
            embedded = self.word_embedding_prj(embedded) # batch x word x hidden
            encoder_output, encoder_hidden = self.encoder_gru(embedded) # encoder_hidden 1/2 x batch x hidden
            encoder_hidden = encoder_hidden.permute(1, 0, 2).reshape(encoder_hidden.shape[1], 1, -1) if \
                            encoder_hidden.shape[0] == 2 else encoder_hidden
            if self.state_hidden is None:
                self.state_hidden = torch.zeros_like(encoder_hidden)
            state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden)
            self.state_hidden = state_hidden.detach()
            value = self.critic(state_output)
            state_hidden = state_hidden.transpose(0, 1).contiguous().squeeze(1) # batch x hidden
            
        elif self.value_network == 'diff':
            diff_output, diff_hidden_ = self.diff_gru(graph_encoding)
            diff_hidden = diff_hidden_.view(batch_size, 1, self.hidden_size * (2 if self.bidirectional else 1))
            value = self.critic(diff_hidden)
            
        elif self.value_network == 'local':
            # Local Graph
            # graph # num_nodes x entities
            localkg_embedded = self.word_embedding(local_graph)  # nodes x entities x hidden+
            localkg_embedded = self.word_embedding_prj(localkg_embedded) #  nodes x  entities x hidden
            localkg_embedded = localkg_embedded.mean(1) # nodes x hidden
            localkg_embedded = torch.stack([localkg_embedded]*batch_size,0) # batch x nodes x hidden
            localkg_encoding = self.localkg_gat(localkg_embedded, local_adj.float())
            localkg_output, localkg_hidden =  self.local_gru(localkg_encoding)
            localkg_hidden = localkg_hidden.view(1, 1, self.hidden_size * (2 if self.bidirectional else 1))
            value = self.critic(localkg_hidden)

        # Commands/Actions
        cmds_embedding = self.word_embedding(commands)
        cmds_embedding = self.word_embedding_prj(cmds_embedding)
        cmds_embedding = cmds_embedding.view(batch_size * nb_cmds, commands.size(2), self.hidden_size)  # [batch-ncmds] x nentities x hidden_size
        _, cmds_encoding = self.dn_gru.forward(cmds_embedding)  # 1/2 x [batch-ncmds] x hidden
        cmds_encoding = cmds_encoding.permute(1, 0, 2).reshape(1, cmds_encoding.shape[1], -1) if \
            cmds_encoding.shape[0] == 2 else cmds_encoding
        cmds_encoding = cmds_encoding.squeeze(0)
        cmds_encoding = cmds_encoding.view(batch_size, nb_cmds, self.rnn_hidden_size * (2 if self.bidirectional else 1))
        
        cmds_representation = torch.stack([cmds_encoding] * nb_items, 1).view(batch_size, nb_items, nb_cmds, self.hidden_size)
        graph_representation = torch.stack([graph_encoding] * nb_cmds, 2).view(batch_size, nb_items, nb_cmds, self.hidden_size)
        cmd_selector_new_input = torch.cat((cmds_representation, graph_representation), 3)
        scores = self.att_cmd(cmd_selector_new_input).squeeze(-1)
        scores = torch.mean(scores, 1)
        probs = masked_softmax(scores, commands.sum(dim=2) > 0, dim=1)
        index = probs.multinomial(num_samples=1).unsqueeze(0)
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size * (2 if self.bidirectional else 1), device=self.device)

    def reset_hidden_per_batch(self, batch_id):
        self.state_hidden[:,batch_id,:] = torch.zeros(1, 1, self.hidden_size * (2 if self.bidirectional else 1), device=self.device)
