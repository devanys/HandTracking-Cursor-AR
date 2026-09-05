# HandTracking-Cursor-AR

https://github.com/user-attachments/assets/c0ea3f9c-e045-4cf6-a1fe-fcce3118e119

https://github.com/user-attachments/assets/8342919c-88c4-497a-89db-46e6486e3653

# 🎯 YOLOv26-AR: Real-Time Object Detection & Hand Pose Control System

This project integrates the cutting-edge **YOLOv26** (You Only Look Once) object detection architecture with real-time hand pose tracking to control system cursor and volume. Unlike traditional two-stage detectors, YOLOv26 frames object detection as a single regression problem, looking at the entire image at once to predict bounding boxes and class probabilities simultaneously. 

The hand skeleton is modeled using a 21-landmark topology, allowing for precise joint angle calculations and gesture recognition.

---

## 🧮 The Mathematics of YOLOv26 & Hand Pose Tracking

The system relies on a complex pipeline of deep learning regression, kinematic geometry, and signal processing. Below is the comprehensive mathematical foundation of the system.

### 1. Grid Division & Bounding Box Regression
The input image is divided into an $S \times S$ grid. Each grid cell predicts $B$ bounding boxes. A bounding box is represented by a 5-tuple: $(x, y, w, h, C)$. 

The confidence score $C$ reflects the accuracy of the box containing a hand:

$$
C = P_r(\text{Hand}) \times \text{IoU}_{\text{pred}}^{\text{truth}}
$$

The predicted coordinates $(x, y, w, h)$ are transformed using sigmoid functions $\sigma$ to constrain centers within grid cells, and exponential functions to scale dimensions relative to prior anchors:

$$
x = \sigma(t_x) + c_x
$$

$$
y = \sigma(t_y) + c_y
$$

$$
w = p_w e^{t_w}
$$

$$
h = p_h e^{t_h}
$$

### 2. Complete Intersection over Union (CIoU) Loss
For bounding box regression, YOLOv26 optimizes using the **CIoU** loss, which accounts for overlap area, distance between center points, and aspect ratio consistency:

$$
\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v
$$

The standard **Intersection over Union (IoU)** evaluates overlap:

$$
IoU = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}
$$

The aspect ratio consistency $v$ and trade-off parameter $\alpha$ are defined as:

$$
v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2
$$

$$
\alpha = \frac{v}{(1 - IoU) + v}
$$

### 3. Focal Loss for Classification
To address the extreme foreground-background class imbalance during dense predictions, YOLOv26 utilizes **Focal Loss**:

$$
\mathcal{L}_{fl}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Where:
* $p_t$ is the model's estimated probability for the correct class.
* $\gamma$ is the focusing parameter (down-weights easy examples).
* $\alpha_t$ is the balancing parameter.

### 4. Non-Maximum Suppression (NMS)
To filter multiple overlapping predictions for the same hand, NMS is applied. Predictions are sorted by confidence score $S$. A box $B_i$ is discarded if its IoU with a higher-scoring box $B_{max}$ exceeds a threshold $N_t$:

$$
D \leftarrow D \setminus \{B_i \mid IoU(B_{max}, B_i) \geq N_t\}
$$

### 5. SiLU Activation Function
The hidden layers in the YOLOv26 backbone utilize the **Sigmoid Linear Unit (SiLU)** activation function, which has been proven to outperform ReLU in deep architectures:

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

### 6. Feature Map Dimensionality (Convolutional Math)
The spatial dimensions of the output feature map after a convolutional layer are determined by:

$$
O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

Where $W$ is input size, $K$ is kernel size, $P$ is padding, and $S$ is stride.

---

## 🖐️ Mathematics of Hand Pose & Kinematics

### 7. Hand Joint Angle Calculation (Cosine Similarity)
To determine if a finger is bent or extended, we calculate the interior angle $\theta$ at the proximal interphalangeal (PIP) joint. Given three consecutive landmarks $A$ (MCP), $B$ (PIP), and $C$ (DIP), the angle is derived using the dot product of vectors $\vec{BA}$ and $\vec{BC}$:

$$
\cos \theta = \frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|}
$$

$$
\theta = \arccos \left( \frac{(x_A - x_B)(x_C - x_B) + (y_A - y_B)(y_C - y_B)}{\sqrt{(x_A - x_B)^2 + (y_A - y_B)^2} \cdot \sqrt{(x_C - x_B)^2 + (y_C - y_B)^2}} \right)
$$

A finger is considered "extended" if $\theta > 160^\circ$.

### 8. Finger Extension Metric (Distance Ratio)
As a redundant metric for gesture robustness, a finger is classified as open if the Euclidean distance from the wrist to the fingertip is greater than the distance from the wrist to the knuckle (MCP) by a threshold factor $k$:

$$
E_{finger} = \frac{\|P_{tip} - P_{wrist}\|_2}{\|P_{mcp} - P_{wrist}\|_2}
$$

If $E_{finger} > k$ (where $k \approx 1.1$), the finger is extended.

### 9. Pinch / Volume Control Distance
The volume control gesture relies on the Euclidean distance $d$ between the thumb tip (landmark 4) and index finger tip (landmark 8):

$$
d = \sqrt{(x_8 - x_4)^2 + (y_8 - y_4)^2}
$$

The rate of change of volume is proportional to the discrete time derivative of this distance:

$$
\Delta V \propto \frac{d_t - d_{t-1}}{\Delta t}
$$

### 10. Exponential Moving Average (EMA) Smoothing
Raw hand landmark coordinates from the camera exhibit high-frequency noise (jitter). To smooth cursor movement, we apply an EMA filter:

$$
S_t = \beta S_{t-1} + (1 - \beta) T_t
$$

Where $S_t$ is the smoothed coordinate, $T_t$ is the raw target coordinate, and $\beta$ is the smoothing factor (e.g., $0.8$).

### 11. Coordinate Mapping (Screen Homography)
To map normalized camera coordinates $(u, v)$ to screen pixels $(X_s, Y_s)$, a linear interpolation bounded by a region of interest (ROI) is used:

$$
X_s = W_{screen} \cdot \max\left(0, \min\left(1, \frac{u - x_{min}}{x_{max} - x_{min}}\right)\right)
$$

$$
Y_s = H_{screen} \cdot \max\left(0, \min\left(1, \frac{v - y_{min}}{y_{max} - y_{min}}\right)\right)
$$
