import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STARTER CODE (UNCHANGED)
# ============================================================================

model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]

imagenet_path = './imagenet_samples'

# FIXED: Get image files directly from the directory
if not os.path.exists(imagenet_path):
    raise FileNotFoundError(f"Directory not found: {imagenet_path}")

# Get all image files directly (not subdirectories)
all_files = os.listdir(imagenet_path)
image_files = [f for f in all_files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

if len(image_files) == 0:
    print(f"No image files found in {imagenet_path}")
    print(f"Contents: {all_files}")
    raise ValueError("Expected image files (.jpg, .jpeg, or .png)")

print(f"Found {len(image_files)} images in {imagenet_path}")

# ============================================================================
# NEW CODE: HELPER FUNCTIONS (FROM SCRATCH)
# ============================================================================

def simple_superpixels(img_array, n_segments=50):
    """Simple grid-based superpixel segmentation"""
    h, w = img_array.shape[:2]
    grid_size = int(np.sqrt(h * w / n_segments))
    segments = np.zeros((h, w), dtype=int)
    label = 0
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            segments[i:i+grid_size, j:j+grid_size] = label
            label += 1
    return segments

def ridge_regression(X, y, weights, alpha=1.0):
    """Ridge regression: beta = (X'WX + aI)^-1 X'Wy"""
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX + alpha * np.eye(X.shape[1]), XtWy)
    return beta

def kendall_tau(x, y):
    """Kendall-Tau correlation"""
    n = len(x)
    concordant = sum((x[i] - x[j]) * (y[i] - y[j]) > 0 
                     for i in range(n) for j in range(i+1, n))
    return 2 * concordant / (n * (n - 1)) - 1

def spearman_rho(x, y):
    """Spearman rank correlation"""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]

# ============================================================================
# NEW CODE: SMOOTHGRAD IMPLEMENTATION
# ============================================================================

def smoothgrad(model, input_tensor, target_class, n_samples=50, noise_level=0.15):
    """SmoothGrad: average gradients over noisy samples"""
    stdev = noise_level * (input_tensor.max() - input_tensor.min())
    total_gradients = torch.zeros_like(input_tensor)
    
    for _ in range(n_samples):
        noisy_input = input_tensor + torch.randn_like(input_tensor) * stdev
        noisy_input.requires_grad = True
        
        output = model(noisy_input)
        model.zero_grad()
        output[0, target_class].backward()
        
        total_gradients += noisy_input.grad.data
    
    return (total_gradients / n_samples).squeeze(0)

# ============================================================================
# NEW CODE: LIME IMPLEMENTATION
# ============================================================================

def lime(model, image_pil, preprocess_fn, target_class, n_samples=1000):
    """LIME: local linear approximation with superpixel perturbations"""
    # Convert to array and get superpixels
    img_array = np.array(image_pil.resize((224, 224))) / 255.0
    segments = simple_superpixels(img_array, n_segments=50)
    n_superpixels = segments.max() + 1
    
    # Generate perturbed samples
    X = np.random.randint(0, 2, (n_samples, n_superpixels))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Create perturbed image
        perturbed = img_array.copy()
        for sp in range(n_superpixels):
            if X[i, sp] == 0:
                perturbed[segments == sp] = 0.5  # Gray out
        
        # Get prediction
        perturbed_pil = Image.fromarray((perturbed * 255).astype(np.uint8))
        perturbed_tensor = preprocess_fn(perturbed_pil).unsqueeze(0)
        with torch.no_grad():
            y[i] = model(perturbed_tensor)[0, target_class].item()
    
    # Fit weighted linear model
    distances = np.sum((X - 1) ** 2, axis=1)
    weights = np.exp(-distances / 0.25**2)
    beta = ridge_regression(X, y, weights, alpha=1.0)
    
    return np.abs(beta), segments

# ============================================================================
# NEW CODE: VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_explanations(img_pil, smoothgrad_exp, lime_exp, segments, img_name):
    """Visualize both explanations side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img_pil)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # SmoothGrad - FIX: resize image to match gradient dimensions
    grad_map = torch.abs(smoothgrad_exp).mean(dim=0).cpu().numpy()
    axes[1].imshow(img_pil.resize((224, 224)))  # ← Changed this line
    axes[1].imshow(grad_map, cmap='hot', alpha=0.6)
    axes[1].set_title('SmoothGrad')
    axes[1].axis('off')
    
    # LIME
    lime_map = np.zeros((224, 224))
    for i in range(len(lime_exp)):
        lime_map[segments == i] = lime_exp[i]
    axes[2].imshow(img_pil.resize((224, 224)))
    axes[2].imshow(lime_map, cmap='hot', alpha=0.6)
    axes[2].set_title('LIME')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'explanations_{img_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    
def compare_methods(smoothgrad_exp, lime_exp, segments):
    """Compute correlation between methods"""
    # Aggregate SmoothGrad per superpixel
    sg_map = torch.abs(smoothgrad_exp).mean(dim=0).cpu().numpy()
    sg_per_sp = np.array([sg_map[segments == i].mean() for i in range(len(lime_exp))])
    
    return kendall_tau(sg_per_sp, lime_exp), spearman_rho(sg_per_sp, lime_exp)

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("=" * 80)
print("EXPLANATION METHODS FOR RESNET-18 ON IMAGENET")
print("=" * 80)

results = []

for img_file in sorted(image_files):
    img_path = os.path.join(imagenet_path, img_file)
    img_name = os.path.splitext(img_file)[0]  # Remove extension for output names
    
    print(f"\n{'='*80}")
    print(f"Image: {img_file}")
    print(f"{'='*80}")
    
    # Load image
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)
    
    pred_idx = top5_idx[0].item()
    pred_label = idx2label[pred_idx]
    pred_prob = top5_prob[0].item()
    
    print(f"\nTop-5 Predictions:")
    for i in range(5):
        print(f"  {i+1}. {idx2label[top5_idx[i].item()]}: {top5_prob[i].item():.3f}")
    
    # Generate explanations
    print("\nGenerating explanations...")
    sg_exp = smoothgrad(model, input_tensor, pred_idx, n_samples=50, noise_level=0.15)
    lime_exp, segments = lime(model, input_image, preprocess, pred_idx, n_samples=1000)
    
    # Compare methods
    kt, sr = compare_methods(sg_exp, lime_exp, segments)
    print(f"\nMethod Correlation:")
    print(f"  Kendall-Tau: {kt:.3f}")
    print(f"  Spearman: {sr:.3f}")
    
    # Qualitative analysis
    print(f"\nAnalysis:")
    print(f"  Confidence: {pred_prob:.1%} ({'High' if pred_prob > 0.7 else 'Medium' if pred_prob > 0.4 else 'Low'})")
    
    sg_max = torch.abs(sg_exp).max().item()
    sg_mean = torch.abs(sg_exp).mean().item()
    print(f"  SmoothGrad: {'Focused' if sg_max/sg_mean > 10 else 'Distributed'} attention")
    
    lime_top5_ratio = lime_exp[np.argsort(lime_exp)[-5:]].sum() / lime_exp.sum()
    print(f"  LIME: Top 5 superpixels = {lime_top5_ratio:.1%} of importance")
    print(f"  Agreement: {'Strong' if kt > 0.5 else 'Moderate' if kt > 0.2 else 'Weak'}")
    
    # Visualize
    visualize_explanations(input_image, sg_exp, lime_exp, segments, img_name)
    print(f"  Saved: explanations_{img_name}.png")
    
    results.append({'image': img_name, 'kendall': kt, 'spearman': sr})

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if len(results) == 0:
    print("No images were processed!")
else:
    kt_vals = [r['kendall'] for r in results]
    sr_vals = [r['spearman'] for r in results]
    print(f"Processed {len(results)} images")
    print(f"Average Kendall-Tau: {np.mean(kt_vals):.3f} ± {np.std(kt_vals):.3f}")
    print(f"Average Spearman: {np.mean(sr_vals):.3f} ± {np.std(sr_vals):.3f}")

    print("""
KEY INSIGHTS:
- SmoothGrad: Pixel-level gradients, captures fine details
- LIME: Region-level importance, more human-interpretable
- Positive correlation: Both identify similar important areas
- Use SmoothGrad for detailed analysis, LIME for communication
""")