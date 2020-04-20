

## SSD from scratch

The following model has been trained on google colab from scratch:

1. [SSD with Multi Box Loss](<https://gitlab.com/entirety.ai/meetup-intuition-to-implementation/blob/master/Phase%20-%202/SSD/SSD_full.ipynb>)

2. [SSD with Focal Loss](<https://gitlab.com/entirety.ai/meetup-intuition-to-implementation/blob/master/Phase%20-%202/SSD/Focal_Loss_SSD.ipynb>)



###  Summary

| Model                                                        | epochs | Change in train loss | Loss Function                    | mAP         | weights                                                      |
| ------------------------------------------------------------ | ------ | -------------------- | -------------------------------- | ----------- | ------------------------------------------------------------ |
| [SSD with Multi Box Loss](<https://gitlab.com/entirety.ai/meetup-intuition-to-implementation/blob/master/Phase%20-%202/SSD/SSD_full.ipynb>) | 160    | 12.853 -> 3.0090     | MultiBox Loss with Cross Entropy | 58.1        | [Google drive](https://drive.google.com/open?id=1-6NBJ2BNi1F3-xnoSgvIOl74U2yspMem) |
| [SSD with Focal Loss](<https://gitlab.com/entirety.ai/meetup-intuition-to-implementation/blob/master/Phase%20-%202/SSD/Focal_Loss_SSD.ipynb>) | 175    | 2.5304 -> 0.803      | MultiBox Loss with Focal Loss    | In progress | [Google drive](https://drive.google.com/open?id=1-56TBgCEIHhtCtdKOXH5yP1_do14F38w) |





### TODO

- Run for 300 epochs to match to research paper results.
- Compare SSD results for different Loss Functions.
- Finetune the model using VOC 2012 dataset
- Play with Batch Norm and Dropout in Auxiliary Convolution
- Multi-GPU training