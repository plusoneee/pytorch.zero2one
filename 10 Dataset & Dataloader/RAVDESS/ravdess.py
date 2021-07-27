import cv2
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RAVDESSVideoDataset(Dataset):
    def __init__(
            self,
            clips_csv_file,
            dataset_dir,
            time_depth,
            channels=3,
            width=1280,
            height=720,
            frame_rate = 1,
            transform=None
    ):
        """
        Args:
            clips_csv_file (string): Path to the clips (csv) file with labels.
            dataset_dir (string): Directory with all the videoes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            channels: Number of channels of frames
            time_depth: Number of frames to be loaded in a sample
            width, height: Dimensions of the frames

        """

        self.clips_df = pd.read_csv(clips_csv_file)
        self.dataset_dir = dataset_dir
        self.channels = channels
        self.time_depth = time_depth
        self.width = width
        self.height = height
        self.transform = transform
        self.frame_rate = frame_rate

    def __len__(self):
        return len(self.clips_df)

    def read_video(self, video_file):
        # open video file
        self.cap = cv2.VideoCapture(video_file)

        # create frames
        frames = torch.FloatTensor(
            self.time_depth,  self.channels, self.height, self.width
        )

        sec = 0
        failed_clip = False

        for t in range(self.time_depth):
            sec = sec + self.frame_rate
            sec = round(sec, 2)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            has_frames, image = self.cap.read()

            if has_frames:
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C x H x W
                frame = torch.from_numpy(image)
                frame = frame.permute(2, 0, 1)
                frames[t, :, :, :] = frame
                # print(frames.shape)
            else:
                failed_clip = True
                break

        return frames, failed_clip

    def __getitem__(self, idx):
        video_file = os.path.join(self.dataset_dir, self.clips_df['name'][idx])
        clips, failed_clip = self.read_video(video_file)

        # to store transformed images
        _transformed_images = []
        for image in clips:
            if self.transform:
                image = self.transform(image)
                _transformed_images.append(image)

        if _transformed_images:
            clips = torch.stack(_transformed_images, dim=0)

        sample = {
            "clip": clips,
            "label": self.clips_df['label'][idx],
            "name": self.clips_df['name'][idx]
        }
        return sample

if __name__ == '__main__':

    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(720),
        transforms.ToTensor()
    ])

    dataset = RAVDESSVideoDataset(
        clips_csv_file = 'actor3.csv',
        dataset_dir = 'datasets',
        time_depth = 5,
        frame_rate= 0.6, # 0.6 sec one frame
        transform=_transform
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # try one batch
    for i, data in enumerate(dataloader):
        clip = data['clip']
        label = data['label']
        # Batch, SeqLen, Channel, H, W
        print(clip.shape)
        break


