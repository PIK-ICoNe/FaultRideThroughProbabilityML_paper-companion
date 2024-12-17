import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
import os


df_all_nodes = pd.read_csv("data_frames/df_all_nodes.csv")
df_all_lines = pd.read_csv("data_frames/df_all_lines.csv")
output_dir = Path("4Pytorch")

def prepare_data4Pytorch(df_all_nodes, df_all_lines, grid_type, output_dir):
    df_lines = df_all_lines[df_all_lines["grid_type"] == grid_type]
    df_nodes = df_all_nodes[df_all_nodes["grid_type"] == grid_type]

    grid_indices_lines = df_lines["grid_index"].unique()

    grid_indices_nodes = df_nodes["grid_index"].unique()

    if np.array_equal(grid_indices_lines, grid_indices_nodes):
        grid_indices = grid_indices_nodes
    else:
        raise ValueError("Grid indices do not match in df_lines and df_nodes")
        
    # fitting scalar

    all_node_types = df_nodes['node_type']
    all_P = df_nodes['P'].fillna(0)
    all_Q = df_nodes['Q'].fillna(0)
    all_B_x_real = df_nodes['B_x_real'].fillna(0)
    all_sum_y_real = df_nodes['sum_y_real']
    all_sum_y_imag = df_nodes['sum_y_imag']
    all_sum_y_shunt_mk_imag = df_nodes['sum_y_shunt_mk_imag']

    numerical_features = np.column_stack([all_P, all_Q, all_B_x_real, all_sum_y_real, all_sum_y_imag, all_sum_y_shunt_mk_imag])

    # Fit the scaler on numerical features only
    node_scaler = StandardScaler()
    node_scaler.fit(numerical_features)

    # Scale the numerical features in the DataFrame
    df_nodes.loc[:, ['P', 'Q', 'B_x_real', 'sum_y_real', 'sum_y_imag', 'sum_y_shunt_mk_imag']] = node_scaler.transform(numerical_features)

    encoder = OneHotEncoder()
    encoder.fit(all_node_types.values.reshape(-1, 1))


    # Combine all edge features for fitting the scaler
    all_y_real = np.concatenate([df_lines['y_real'], df_lines['y_real']])
    all_y_imag = np.concatenate([df_lines['y_imag'], df_lines['y_imag']])
    all_y_shunt_mk_imag = np.concatenate([df_lines['y_shunt_mk_imag'], df_lines['y_shunt_mk_imag']])
    all_power_flow_P = np.concatenate([df_lines['power_flow_P_ij'], df_lines['power_flow_P_ji']])
    all_power_flow_Q = np.concatenate([df_lines['power_flow_Q_ij'], df_lines['power_flow_Q_ji']])

    edge_features = np.column_stack([all_y_real, all_y_imag, all_y_shunt_mk_imag, all_power_flow_P, all_power_flow_Q])

    # Fit the scaler on all edge features
    edge_scaler = StandardScaler()
    edge_scaler.fit(edge_features)

    # Scale the edge features in the DataFrame
    scaled_edge_features = edge_scaler.transform(edge_features)
    num_edges = len(df_lines)
    df_lines.loc[:,'y_real'] = scaled_edge_features[:num_edges, 0]
    df_lines.loc[:,'y_imag'] = scaled_edge_features[:num_edges, 1]
    df_lines.loc[:,'y_shunt_mk_imag'] = scaled_edge_features[:num_edges, 2]
    df_lines.loc[:,'power_flow_P_ij'] = scaled_edge_features[:num_edges, 3]
    df_lines.loc[:,'power_flow_P_ji'] = scaled_edge_features[num_edges:, 3]
    df_lines.loc[:,'power_flow_Q_ij'] = scaled_edge_features[:num_edges, 4]
    df_lines.loc[:,'power_flow_Q_ji'] = scaled_edge_features[num_edges:, 4]

    output_dir = output_dir / grid_type
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(str(output_dir) + '/input_data.h5', 'w') as h5file:
        # encoder = OneHotEncoder()
        for grid_index in grid_indices:
            df_lines_subset = df_lines[df_lines["grid_index"] == grid_index]
            df_nodes_subset = df_nodes[df_nodes["grid_index"] == grid_index]
            
            edge_index = np.concatenate([df_lines_subset.destination.values, df_lines_subset.source.values]), np.concatenate([df_lines_subset.source.values, df_lines_subset.destination.values])
            
            # node features
            # one hot encoding of the three node types 'PowerDynamics.NormalForm{1}', 'PowerDynamics.PQAlgebraic', 'PowerDynamics.SlackAlgebraic'
            one_hot_encoded = encoder.transform(df_nodes_subset['node_type'].values.reshape(-1, 1))
            node_type_encoded = one_hot_encoded.toarray()
            P = df_nodes_subset.P.values.reshape(-1, 1)
            Q = df_nodes_subset.Q.values.reshape(-1, 1)
            B_x_real = df_nodes_subset['B_x_real'].values.reshape(-1, 1)
            sum_y_real = df_nodes_subset.sum_y_real.values.reshape(-1, 1)
            sum_y_imag = df_nodes_subset.sum_y_imag.values.reshape(-1, 1)
            sum_y_shunt_mk_imag = df_nodes_subset.sum_y_shunt_mk_imag.values.reshape(-1, 1)
            node_features = np.hstack([node_type_encoded, P, Q, B_x_real, sum_y_real, sum_y_imag, sum_y_shunt_mk_imag])
            
            # edge_features
            y_real = np.concatenate([df_lines_subset.y_real, df_lines_subset.y_real])
            y_imag = np.concatenate([df_lines_subset.y_imag, df_lines_subset.y_imag])
            y_shunt_mk_imag = np.concatenate([df_lines_subset.y_shunt_mk_imag, df_lines_subset.y_shunt_mk_imag])
            power_flow_P = np.concatenate([df_lines_subset.power_flow_P_ij, df_lines_subset.power_flow_P_ji])
            power_flow_Q = np.concatenate([df_lines_subset.power_flow_Q_ij, df_lines_subset.power_flow_Q_ji])
            edge_features = np.column_stack([y_real, y_imag, y_shunt_mk_imag, power_flow_P, power_flow_Q])
            
            # mask, to avoid evaluating the losses at the slack busses
            mask = df_nodes_subset.node_type != 'PowerDynamics.SlackAlgebraic'
            mask = mask.values.astype(np.uint8)  # Convert to uint8 for storage
            
            #targets
            targets_surv = df_nodes_subset["surv"]
            targets_snbs = df_nodes_subset["snbs"]
            
            group = h5file.create_group(str(grid_index))
            group.create_dataset('node_features', data=node_features)
            group.create_dataset('edge_index', data=edge_index)
            group.create_dataset('edge_features', data=edge_features)
            group.create_dataset('mask', data=mask)
            target_group = group.create_group('targets')
            target_group.create_dataset('surv', data=targets_surv)
            target_group.create_dataset('snbs', data=targets_snbs)        

prepare_data4Pytorch(df_all_nodes, df_all_lines, "ieee", output_dir)
prepare_data4Pytorch(df_all_nodes, df_all_lines, "synthetic", output_dir)