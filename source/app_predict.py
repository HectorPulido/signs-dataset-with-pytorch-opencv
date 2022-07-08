import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


class Net(nn.Module):
    def __init__(self, num_channels):
        super(Net, self).__init__()

        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(
            self.num_channels, self.num_channels * 2, 3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        self.conv3 = nn.Conv2d(
            self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(self.num_channels * 4)

        self.fc1 = nn.Linear(self.num_channels * 4 * 8 * 8, self.num_channels * 4)
        self.bn4 = nn.BatchNorm1d(self.num_channels * 4)
        self.fc2 = nn.Linear(self.num_channels * 4, 6)

    def forward(self, x):
        # 3 x 64 x 64
        x = self.conv1(x)
        x = self.bn1(x)
        # num_channels x 64 x 64
        x = F.max_pool2d(x, 2)
        # num_channels x 32 x 32
        x = F.relu(x)
        # x = F.dropout2d(x, 0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        # num_channels * 2 x 32 x 32
        x = F.max_pool2d(x, 2)
        # num_channels * 2 x 16 x 16
        x = F.relu(x)
        # x = F.dropout2d(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        # num_channels * 4 x 16 x 16
        x = F.max_pool2d(x, 2)
        # num_channels * 4 x 4 x 4
        x = F.relu(x)
        # x = F.dropout2d(x, 0.2)

        x = x.view(-1, self.num_channels * 4 * 8 * 8)

        # self.num_channels * 4 * 8 * 8
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        # x = F.dropout(x, 0.2)

        # self.num_channels * 4
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        # 6
        return x

    @staticmethod
    def load_model(num_channels, model_name, device):
        net = Net(num_channels).to(device)
        net.load_state_dict(torch.load(model_name))
        net.eval()
        return net

    @staticmethod
    def transform():
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def set_image_square(frame):
        w = frame.shape[0] if frame.shape[0] < frame.shape[1] else frame.shape[1]
        h = frame.shape[0] if frame.shape[0] < frame.shape[1] else frame.shape[1]
        x = frame.shape[1] / 2 - w / 2
        y = frame.shape[0] / 2 - h / 2

        frame = frame[int(y) : int(y + h), int(x) : int(x + w)]
        return frame

    @staticmethod
    def apply_transform(frame, transform, device):
        dim = (64, 64)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_NEAREST)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return transform(resized).reshape(1, 3, 64, 64).to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = Net.transform()
    net = Net.load_model(64, "model", device)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame = Net.set_image_square(frame)

        data = Net.apply_transform(frame, transforms, device)
        pred = torch.argmax(net.forward(data))

        image = cv2.putText(
            frame,
            str(int(pred)),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            5,
            cv2.LINE_AA,
        )

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
