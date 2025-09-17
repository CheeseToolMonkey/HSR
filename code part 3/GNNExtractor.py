###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard libraries
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

torch.autograd.set_detect_anomaly(True)

# GNNs and GRNNs
import GNNs.Utils.graphML as gml
import GNNs.Modules.architectures as archit
import GNNs.Modules.architecturesTime as architTime

# Stable Baselines
from stable_baselines3.common.utils import get_device

###############################################################################
############################# G N N   A G E N T ###############################
###############################################################################
class BaseGNNExtractor(object):
    """
    Constructs a GNN that receives observations as an input (including a graph shift operator, GSO) 
    and outputs a (latent) representation for policy and value networks.
    Arguments:
        S: GSO
        hidden_dim: dimension of the hidden state
        batch_size: number of simultaneous realizations
        p: dimension of local decision variables (# first layer features)
        lr: learning rate (meta problem)
        device: processing unit (either 'cpu' or 'cuda:0')
    """

    def __init__(self, hidden_dim, batchsize, n_agents, S, p, action_dim,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ):
        super(BaseGNNExtractor, self).__init__()

        self.device = get_device(device)

        # GRNN architecture hyperparams
        F1 = hidden_dim  # number of state features
        K1 = 4  # number of filter taps for the first layer
        self.F1 = F1
        self.K1 = K1
        self.batch_size = batchsize
        self.n_feats_input = p
        self.n_agents = n_agents
        self.graph_dim = n_agents**2
        self.output_features = action_dim

        # Hidden states
        self.h_t = torch.zeros(self.batch_size, self.F1, self.n_agents)
        # Defining the GRNN
        self.S = S  # GSO
        self.n_params = 0

    def trainer_builder(self, grnn_archit, beta1, beta2):
        trainer = optim.Adam(grnn_archit.parameters(), self.trainer_lr, betas=(beta1, beta2))

        return trainer

    def reset_hidden_state(self):
        self.h_t = torch.zeros(self.batch_size, self.F1, self.n_agents)

    def reuse_hidden_state(self):
        # detaching hidden state from the comp graph
        h_t = Variable(self.h_t.data, requires_grad=True)
        self.h_t = h_t.detach()


class TEGCRNNExtractor(BaseGNNExtractor):

    def __init__(self, hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action,
                 device='cpu', gnn_layers=3*[10], gnn_feats=3*[5]):

        super().__init__(hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action,
        n_feats_action, net_arch = [dict(pi=[64, 64], vf=[64, 64])], activation_fn=nn.Tanh, device=device)

        hParamsTimeEdgeGCRNN_MLP = {}  # Hyperparameters (hParams) 
        hParamsTimeEdgeGCRNN_MLP['name'] = 'TimeEdgeGCRNNMLP'  # Name of the architecture
        hParamsTimeEdgeGCRNN_MLP['inFeatures'] = n_feats_input
        hParamsTimeEdgeGCRNN_MLP['stateFeatures'] = self.F1
        hParamsTimeEdgeGCRNN_MLP['inputFilterTaps'] = self.K1
        hParamsTimeEdgeGCRNN_MLP['stateFilterTaps'] = self.K1
        hParamsTimeEdgeGCRNN_MLP['stateNonlinearity'] = nn.Tanh
        hParamsTimeEdgeGCRNN_MLP['outputNonlinearity'] = nn.Identity
        hParamsTimeEdgeGCRNN_MLP['dimLayersMLP'] = [self.output_features]
        hParamsTimeEdgeGCRNN_MLP['bias'] = True
        hParamsTimeEdgeGCRNN_MLP['time_gating'] = True
        hParamsTimeEdgeGCRNN_MLP['spatial_gating'] = 'edge'
        hParamsTimeEdgeGCRNN_MLP['mlpType'] = 'multipMlp'

        self.pol_archit, self.value_archit = self.archit_builder(hParamsTimeEdgeGCRNN_MLP, S)
        # Storing learnable parameters in the device
        self.pol_archit.to(device)
        self.value_archit.to(device)

        # ADAM optimizer (meta problem, i.e., optimizing the GRNN params)
        self.trainer_lr = lr
        beta1 = 0.9
        beta2 = 0.999
        self.trainer = self.trainer_builder(self.grnn_archit, beta1, beta2)

    def archit_builder(self, hyperparams, S):
        pol_archit = archit.GatedGCRNNforRegression(hyperparams['inFeatures'],
                                                    hyperparams['stateFeatures'],
                                                    hyperparams['inputFilterTaps'],
                                                    hyperparams['stateFilterTaps'],
                                                    hyperparams['stateNonlinearity'],
                                                    hyperparams['outputNonlinearity'],
                                                    hyperparams['dimLayersMLP'],
                                                    S,
                                                    hyperparams['bias'],
                                                    hyperparams['time_gating'],
                                                    hyperparams['spatial_gating'],
                                                    hyperparams['mlpType'])

        value_archit = archit.GatedGCRNNforRegression(hyperparams['inFeatures'],
                                                    hyperparams['stateFeatures'],
                                                    hyperparams['inputFilterTaps'],
                                                    hyperparams['stateFilterTaps'],
                                                    hyperparams['stateNonlinearity'],
                                                    hyperparams['outputNonlinearity'],
                                                    hyperparams['dimLayersMLP'],
                                                    S,
                                                    hyperparams['bias'],
                                                    hyperparams['time_gating'],
                                                    hyperparams['spatial_gating'],
                                                    hyperparams['mlpType'])
        # Counting learnable parameters --- from GRNN og code
        for param in list(opt_archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.n_params += thisNParam
            else:
                pass

        # Initialize hidden states
        self.h_t.detach_()
        self.h_t = torch.zeros(self.batch_size, self.F1, self.n_agents)

        return opt_archit

    # TODO: customize forward method to take GSO into account
    def forward(self, features: torch.Tensor, h_t1_value, h_t1_pol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        S_t = features[:, :self.graph_dim].reshape(self.n_agents, self.n_agents)
        gnn_input = features[:, self.graph_dim:].reshape(self.n_agents, self.n_feats_input)
        self.value_archit.S = S_t
        self.value_archit.S = self.value_archit.S.to(self.device)
        self.pol_archit.S = S_t
        self.pol_archit.S = self.pol_archit.S.to(self.device)
        value_latent, h_t_value = self.value_archit(gnn_input, h_t1_value)
        policy_latent, h_t_pol = self.pol_archit(gnn_input, h_t1_value)
        return policy_latent, h_t_pol, value_latent, h_t_value


# Selection GNN
class SelGNNExtractor(BaseGNNExtractor):

    def __init__(self, hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action, device='cuda:0', net_arch=[],
                 activation_fn=nn.Tanh, gnn_layers=2*[3], gnn_feats=3*[64], lr=1e-3):
        super().__init__(hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action,
                         n_feats_action, net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=nn.Tanh,
                         device=device)

        hParamsSelGNNDeg = {}  # Hyperparameters (hParams) for the Selection GNN (SelGNN)
        hParamsSelGNNDeg['name'] = 'Sel'  # Name of the architecture
        inFeatures = n_feats_input
        hParamsSelGNNDeg['F'] = [inFeatures] + gnn_feats #+ [self.output_features]  # Features per layer
        hParamsSelGNNDeg['Fvalue'] = [inFeatures] + gnn_feats #+ [self.output_features]  # Features per layer / value network
        hParamsSelGNNDeg['K'] = [self.K1] + gnn_layers  # Number of filter taps per layer
        hParamsSelGNNDeg['bias'] = True  # Decide whether to include a bias term
        hParamsSelGNNDeg['sigma'] = nn.Tanh # Selected nonlinearity
        hParamsSelGNNDeg['N'] = [self.n_agents, self.n_agents, self.n_agents]  # Number of nodes at the end of each layer
        hParamsSelGNNDeg['rho'] = gml.NoPool  # Summarizing function
        hParamsSelGNNDeg['alpha'] = [1] + [1]*len(gnn_layers)  # alpha-hop neighborhood
        hParamsSelGNNDeg['valueoutputNonlinearity'] = nn.ReLU
        # hParamsSelGNNDeg['dimLayersMLP'] = [self.output_features]
        hParamsSelGNNDeg['dimLayersMLP'] = []  #[self.output_features]
        hParamsSelGNNDeg['mlpType'] = 'oneMlp'

        self.policy_net, self.value_net = self.archit_builder(hParamsSelGNNDeg, S)
        self.latent_dim_pi = gnn_feats[-1]
        self.latent_dim_vf = gnn_feats[-1]
        # Storing learnable parameters in the device
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        # ADAM optimizer (meta problem, i.e., optimizing the GRNN params)
        self.trainer_lr = lr
        beta1 = 0.9
        beta2 = 0.999
        # self.trainer = self.trainer_builder(self.grnn_archit, beta1, beta2)

    def archit_builder(self, hyperparams, S):
        pol_archit = archit.SelectionGNN(  # Graph filtering
            hyperparams['F'],
            hyperparams['K'],
            hyperparams['bias'],
            # Nonlinearity
            hyperparams['sigma'],
            # Pooling
            hyperparams['N'],
            hyperparams['rho'],
            hyperparams['alpha'],
            # MLP
            hyperparams['dimLayersMLP'],
            # Structure
            S)

        value_archit = archit.SelectionGNN(  # Graph filtering
            hyperparams['Fvalue'],
            hyperparams['K'],
            hyperparams['bias'],
            # Nonlinearity
            hyperparams['sigma'],
            # Pooling
            hyperparams['N'],
            hyperparams['rho'],
            hyperparams['alpha'],
            # MLP
            hyperparams['dimLayersMLP'],
            # Structure
            S)

        # Counting learnable parameters --- from GRNN og code
        for param in list(pol_archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.n_params += thisNParam
            else:
                pass

        # Initialize hidden states
        self.h_t.detach_()
        self.h_t = torch.zeros(self.batch_size, self.F1, self.n_agents)

        return pol_archit, value_archit

    # TODO: customize forward method to take GSO into account
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        S_t = features[:, :self.graph_dim].reshape(self.n_agents, self.n_agents)
        gnn_input = features[:, self.graph_dim:].reshape(self.n_agents, self.n_feats_input)
        self.value_net.S = S_t
        self.value_net.S = self.value_net.S.to(self.device)
        self.policy_net.S = S_t
        self.policy_net.S = self.policy_net.S.to(self.device)
        value_latent = self.value_net(gnn_input)
        policy_latent = self.policy_net(gnn_input)
        return policy_latent, value_latent


# Accounting for time-varying GSOs
class TDBGNNExtractor(BaseGNNExtractor):

    def __init__(self, hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action, device='cuda:0', net_arch=[],
                 nonlinearity=nn.Tanh, gnn_layers=2*[4], gnn_feats=3*[32], lr=1e-3):
        super().__init__(hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action,
                         n_feats_action, net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=nn.Tanh,
                         device=device)

        hParamsLocalGNN = {}
        hParamsLocalGNN['name'] = 'LocalGNN'
        # Chosen architecture
        hParamsLocalGNN['archit'] = architTime.LocalGNN_B
        hParamsLocalGNN['device'] = device
        # Graph convolutional parameters
        hParamsLocalGNN['dimNodeSignals'] = [n_feats_input] + gnn_feats  # Features per layer
        hParamsLocalGNN['nFilterTaps'] = [self.K1] + gnn_layers  # Number of filter taps
        hParamsLocalGNN['bias'] = True  # Decide whether to include a bias term
        # Nonlinearity
        hParamsLocalGNN['nonlinearity'] = nonlinearity  # Selected nonlinearity
        # is affected by the summary
        # Readout layer: local linear combination of features
        hParamsLocalGNN['dimReadout'] = [32, n_feats_action]  # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor
        # considering all nodes at once, making the architecture entirely
        # local.
        # Graph structure
        hParamsLocalGNN['dimEdgeFeatures'] = 1  # Scalar edge weights

        self.policy_net, self.value_net = self.archit_builder(hParamsLocalGNN, S)
        self.latent_dim_pi = hParamsLocalGNN['dimReadout'][-1]*self.n_agents
        self.latent_dim_vf = hParamsLocalGNN['dimReadout'][-1]*self.n_agents
        # Storing learnable parameters in the device
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        # ADAM optimizer (meta problem, i.e., optimizing the GRNN params)
        self.trainer_lr = lr
        beta1 = 0.9
        beta2 = 0.999
        # self.trainer = self.trainer_builder(self.grnn_archit, beta1, beta2)

    def archit_builder(self, hyperparams, S):
        pol_archit = architTime.LocalGNN_B(  # Graph filtering
            hyperparams['dimNodeSignals'],
            hyperparams['nFilterTaps'],
            hyperparams['bias'],
            # Nonlinearity
            hyperparams['nonlinearity'],
            # MLP
            hyperparams['dimReadout'],
            hyperparams['dimEdgeFeatures'])

        value_archit = architTime.LocalGNN_B(  # Graph filtering
            hyperparams['dimNodeSignals'],
            hyperparams['nFilterTaps'],
            hyperparams['bias'],
            # Nonlinearity
            hyperparams['nonlinearity'],
            # MLP
            hyperparams['dimReadout'],
            hyperparams['dimEdgeFeatures'])

        # Counting learnable parameters --- from GRNN og code
        for param in list(pol_archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.n_params += thisNParam
            else:
                pass

        # Initialize hidden states (if using a GRNN)
        self.h_t.detach_()
        self.h_t = torch.zeros(self.batch_size, self.F1, self.n_agents)

        return pol_archit, value_archit


class TDBPGGNNExtractor(BaseGNNExtractor):

    def __init__(self, hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action, device='cuda:0', net_arch=[],
                 nonlinearity=nn.Tanh, gnn_layers=2*[4], gnn_feats=3*[32], lr=1e-3):
        super().__init__(hidden_dim, batchsize, n_agents, S, n_feats_input, n_feats_action,
                         n_feats_action, net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=nn.Tanh,
                         device=device)

        hParamsLocalGNN = {}
        hParamsLocalGNN['name'] = 'LocalGNN'
        hParamsLocalGNN['archit'] = architTime.LocalGNN_B
        hParamsLocalGNN['device'] = device
        hParamsLocalGNN['dimNodeSignals'] = [n_feats_input] + gnn_feats  # Features per layer
        hParamsLocalGNN['nFilterTaps'] = [self.K1] + gnn_layers  # Number of filter taps
        hParamsLocalGNN['bias'] = True
        hParamsLocalGNN['nonlinearity'] = nonlinearity
        hParamsLocalGNN['dimReadout'] = [32, n_feats_action]  # Dimension of the fully connected
        # layers after the GCN layers (map); this fully connected layer
        # is applied only at each node, without any further exchanges nor
        # considering all nodes at once, making the architecture entirely
        # local.
        # Graph structure
        hParamsLocalGNN['dimEdgeFeatures'] = 1  # Scalar edge weights

        self.policy_net = self.archit_builder(hParamsLocalGNN, S)
        self.latent_dim_pi = hParamsLocalGNN['dimReadout'][-1]*self.n_agents
        self.latent_dim_vf = hParamsLocalGNN['dimReadout'][-1]*self.n_agents
        # Storing learnable parameters in the device
        self.policy_net.to(self.device)

        # ADAM optimizer (meta problem, i.e., optimizing the GRNN params)
        self.trainer_lr = lr
        beta1 = 0.9
        beta2 = 0.999
        # self.trainer = self.trainer_builder(self.grnn_archit, beta1, beta2)

    def archit_builder(self, hyperparams, S):
        pol_archit = architTime.LocalGNN_B(  # Graph filtering
            hyperparams['dimNodeSignals'],
            hyperparams['nFilterTaps'],
            hyperparams['bias'],
            # Nonlinearity
            hyperparams['nonlinearity'],
            # MLP
            hyperparams['dimReadout'],
            hyperparams['dimEdgeFeatures'])

        # Counting learnable parameters --- from GRNN og code
        for param in list(pol_archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.n_params += thisNParam
            else:
                pass

        # Initialize hidden states (if using a GRNN)
        self.h_t.detach_()
        self.h_t = torch.zeros(self.batch_size, self.F1, self.n_agents)

        return pol_archit

