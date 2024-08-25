# chalktracks

![Header image - children playing with dump trucks](doc/header_img.webp)

Building a toy dump truck that can follow lines drawn in chalk.

## About

🚂 Toy trains are fun, but your ideas are limited by how much track you have.

🖍️ What if you could draw the tracks with chalk? 

Introducing 🎉 Chalk Tracks! 🎉

🚚 This project aims to build a toy dump truck that follows tracks drawn in chalk. It will drive where you draw!

We're building this in order to:
* Explore a product idea
* Develop an ML-based portfolio project
* Keep the kids entertained!

The evisaged system consists of a motorised toy dump truck with a forward-facing camera fitted, running a segmentation model for chalk line detection. The bulk of the work will involve building the platform, and developing the chalk line segmentation model.

<img src="doc/20240627_172255.jpg" width="400"/> <img src="doc/20240707_152512.jpg" width="400"/> 

**Figures:** First prototype truck, with camera taped on. 
<br/>

https://github.com/user-attachments/assets/9ad01d09-88ea-4e09-aedb-314ebd87270b

**Video:** First prototype chalk segmentation, video captured on a hand-held smartphone, segmentation model trained on [roboflow](https://roboflow.com/).

### Contributors
<a href="https://github.com/chalktracks/chalktracks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=chalktracks/chalktracks" />
</a>

## Segmentation Model Training Workflow

**WIP**

This section describes the workflow for training the segmentation model. 

Code will be added and steps will be documented as the project is built out.

1) Data collection

    Drive the truck over a variety of chalk lines, while recording from the camera

1) Anonymisation 

    Where the dataset contains images of people, blur their faces. Note, no imagery of people will be captured/shared without consent.

1) Save sample videos

    For later demonstration of the segmentation, set aside some video sequences as desired

1) Key framing

    To reduce labelling workload, filter to a keyframe sequence where images are removed if they are too similar to the previous keyframe.

1) Labelling

    Manually annotate the dataset with chalk line lables. Here a simple labelling tool has been developed to annotate the chalk lines with a touch screen interface. After annotation, labels must be converted to appropriate format for training.

1) Split into train/test/validation sets

1) Train the model

1) Convert model for target architecture

1) Deploy/test





