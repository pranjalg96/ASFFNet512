import torch
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the VGGF16 pre-trained model (Should replace with VGGFace)
vgg16_model = models.vgg16(pretrained=True).to(device)

# Set the model to evaluation mode (no training)
vgg16_model.eval()

# MSE Loss
mse_loss = torch.nn.MSELoss(reduction='mean')

# These are locations of max-pool layers in VGG16
max_pool_locs = [4, 9, 16, 23]


def perceptual_loss(restored_img, gt_image):
    """
    Define the perceptual loss function
    :param: restored_img - The restored image tensor
    :param: gt_img - The ground truth image tensor

    returns
    perc_loss - Perceptual loss
    """
    perc_loss = 0
    for layer in max_pool_locs:  
        restored_features = vgg16_model.features[: layer](restored_img)
        gt_features = vgg16_model.features[: layer](gt_image)

        perc_loss += mse_loss(restored_features, gt_features)

    return perc_loss


def gram_matrix(feature_map):
    """
    Calculate Gram matrix of a tensor
    :param: feature_map- A tensor representing a feature map

    returns
    gram - Gram matrix of feature_map
    """

    # Reshape the feature map from (batch_size, num_channels, height, width) to (batch_size, num_channels, -1)
    batch_size, num_channels, height, width = feature_map.size()
    reshaped_feature_map = feature_map.view(batch_size, num_channels, -1)

    gram_matrix = torch.matmul(reshaped_feature_map, reshaped_feature_map.transpose(1, 2))

    # Normalize the Gram matrix by dividing by the number of elements in each feature map
    num_elements = height * width
    gram_matrix = gram_matrix / num_elements

    return gram_matrix


def style_loss(restored_img, gt_image):
    """
    Define the Style loss function
    :param: restored_img - The restored image tensor
    :param: gt_img - The ground truth image tensor

    returns
    loss - Style loss
    """
    loss = 0
    for layer in max_pool_locs:
        restored_features = vgg16_model.features[: layer](restored_img)
        gt_features = vgg16_model.features[: layer](gt_image)

        restored_features_gram = gram_matrix(restored_features)
        gt_features_gram = gram_matrix(gt_features)

        loss += mse_loss(restored_features_gram, gt_features_gram)

    return loss