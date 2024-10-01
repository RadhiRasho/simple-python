import torch
import cv2

model_type = "MiDaS_small"  # Use a smaller model for faster inference

# Load the MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Select device
device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")

# Move model to device and set to evaluation mode
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Select appropriate transform
transform = midas_transforms.small_transform

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to reduce computational load
    frame = cv2.resize(frame, (320, 240))

    # Convert the frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transforms and move to device
    input_batch = transform(img).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert prediction to NumPy array
    output = prediction.cpu().numpy()

    # Normalize the output for display
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply the red and blue colormap
    output_colormap = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    # Display the output
    cv2.imshow("Depth Map", output_colormap)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
