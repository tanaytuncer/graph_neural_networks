import numpy as np

def accuracy(model, data, set = "Train"):
    if fit.pytorch_geometric_implementation:
        h = model(data.x, data.edge_index)
    else:
        h = model(data)
    
    h = h.argmax(dim = 1)
    
    if set == "Train":
        mask = data.train_mask
    elif set == "Validation":
        mask = data.val_mask
    elif set == "Test": 
        mask = data.test_mask

    true_pred = (h[mask] == data.y[mask]).sum()
    model_accuracy = true_pred / len(data.y[mask])
    model_accuracy = model_accuracy.numpy()
    return model_accuracy

def graph_representation(data, smpl_size = 500):
    
    import networkx as nx
    from torch_geometric.data import Data
    from torch_geometric.utils.convert import to_networkx
    from pyvis.network import Network
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    sample_data = Data(data.x[:smpl_size], edge_index = data.edge_index[:smpl_size])
    G = to_networkx(sample_data)
    class_names = data.y[list(G.nodes)].numpy()

    nx.draw(G, 
            with_labels = False,
            node_shape = "o",
            cmap = plt.get_cmap("plasma"),
            alpha = 0.35,
            node_color = class_names,
            node_size = 35,
            edge_color = "grey",
            width = 0.2
            )
    
    plt.axis("off")

def tsne_representation(x, y, title):

    from sklearn.manifold import TSNE
    import plotly_express as px
    tsne = TSNE(n_components = 2, learning_rate=20, random_state = 2023).fit_transform(x.detach().cpu().numpy())

    fig = px.scatter( 
        x = tsne[:,0],
        y = tsne[:,1],
            color = y.numpy().astype(str),
        color_discrete_sequence=["#005200", "#00FF00", "#0000BD", "#0000FF", "#F20000", "#BD0000", "#9C1FBF"],
        opacity = 0.85,
        title = title,
        template="simple_white", 
        width = 500,
        height=460
        )

    fig.update_layout(
        font_family="Times New Roman", 
        title={
        "y":0.95,
        "x":0.5,
        "xanchor": "center",
        "yanchor": "top"}
        )

    fig.update_traces(
        marker = dict(
            symbol = "circle",
            size = 5,
            line = dict(
                width = 0.3,
                color = "DarkSlateGrey")),
            selector=dict(mode="markers")
    )

    fig.update_yaxes(title_text = "", visible = False)
    fig.update_xaxes(title_text = "", visible = False)
    
    fig.show()

def line_chart(x, y, y_label, z = ["#03658C"]):
    import plotly_express as px
    """
    Plot line chart.
        
    x = categorical variable
    y = numerical variable 
    z = color_code
        
    """
    fig = px.line(
        x = x, 
        y = y,
        color_discrete_sequence = z,
        template="simple_white",
        width=800,        
        height=400,
        log_x = False
        )
    
    fig.update_yaxes(title_text = y_label, visible = True)
    fig.update_xaxes(title_text = "Epoch", visible = True)

    return fig

def bar_chart(x, y, title, names):
    import plotly_express as px
    """
    Plot bar chart.
        
    df = dataframe
    x = categorical variable
    y = numerical variable 
        
    """
    fig = px.bar(
        x = x,
        y = y,
        labels = names,
        color_discrete_sequence=["#03658C"],
        template="simple_white",
        text = y,
        width=1000,        
        height=400
    )
        
    fig.update_layout(
        font=dict(family="Times New Roman",size=12),
        yaxis={"categoryorder":"max ascending"},
        title={
        "text": title,
        "y":0.98,
        "x":0.5,
        "xanchor": "center",
        "yanchor": "top"}
        )
        
    fig.update_yaxes(title_text = "", visible = False)
    fig.update_xaxes(title_text = "", visible = False)
        
    return fig.show()

def fit(model, data, n_epochs, lr, weight_decay, pytorch_geometric_implementation = True):

    import torch 
    import tqdm
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    from torch_geometric.data import DataLoader 

    fit.pytorch_geometric_implementation = pytorch_geometric_implementation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    model_performance = []
    model_train_accuracy = []
    model_val_accuracy = []
    model_test_accuracy = []
    criterion = CrossEntropyLoss()
    optimizer = Adam(params = model.parameters(), lr = lr, weight_decay = weight_decay)
    
    with tqdm.trange(n_epochs) as epochs:
        for e in epochs:
            data = data.to(device)

            if fit.pytorch_geometric_implementation:
                model_output = model(data.x, data.edge_index)
            else:   
                model_output = model(data)
            
            optimizer.zero_grad()
            
            compute_loss = criterion(model_output[data.train_mask], data.y[data.train_mask])
            compute_loss.backward()
            optimizer.step()

            train_accuracy = train(model, data)
            val_accuracy = validate(model, data)
            test_accuracy = predict(model, data)
            model_train_accuracy.append(train_accuracy)
            model_val_accuracy.append(val_accuracy)
            model_test_accuracy.append(test_accuracy)

            if e % (n_epochs // 10) == 0:
                model_performance.append(compute_loss.item())
                #print(f"Epoch {e+10}, Training Accuracy: {train_accuracy:.5f}, Validation Accuracy: {val_accuracy:.5f}, Test Accuracy: {test_accuracy:.5f}, Cross Entropy Loss: {compute_loss:.5f}")

    return model_performance, model_train_accuracy, model_val_accuracy, model_test_accuracy

def train(model, data):
    model.eval()
    train_accuracy = accuracy(model, data, set = "Train")
    return train_accuracy

def validate(model, data):
    model.eval()
    val_accuracy = accuracy(model, data, set = "Validation")
    return val_accuracy

def predict(model, data):
    model.eval()
    test_accuracy = accuracy(model, data, set = "Test")
    return test_accuracy
    
def ttest(a, b):
    from scipy import stats

    mean_a = np.mean(a)
    std_dev_a = np.std(a)

    mean_b = np.mean(b)
    std_dev_b = np.std(b)

    t_statistic_ab, p_value_ab = stats.ttest_rel(a, b)
    print(f"T-statistics: {np.round(t_statistic_ab, 3)}, P-value: {np.round(p_value_ab,3)}")

    return t_statistic_ab, p_value_ab, mean_a, std_dev_a, mean_b, std_dev_b