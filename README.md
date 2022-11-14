# The-Beaty-of-Repetition-in-Machine-Composition-Scenarios
This document is designed to assist with reading supplementary materials of the paper “The Beauty of Repetition in Machine Composition Scenarios”. 

In this document, we describe a dataset named Music Repetition Dataset (MRD) based on the Pop piano dataset [1] with the labels of motif-level repetitions. We also describe the code of a novel repetition generation model named Repetition Transformer (R-Transformer) that can generate a large amount of motif-level repetition of different types. Moreover, we demonstrate that the proposed model can generate very interesting results on the motif from pop music and the “fate motif” from classical music. Supplementary materials contain three parts: dataset, code and music demonstration.


## DATASET

In this section, we describe information of Music Repetition Dataset (MRD) in detail. This dataset is extracted based on 1,748 complete pop piano songs from [1]. All songs are in 4/4 time signature, and each song is converted into a symbolic sequence following transcription, synchronization, quantization, and analysis. This dataset contains 562,563 training data and 21,766 test data with the labels of five motif level repetitions. Please see MRD description (./MRD_descrption.pdf) for a more specific description of MRD. The dataset is available at https://connectpolyu-my.sharepoint.com/:f:/g/personal/19045203r_connect_polyu_hk/EiZ6wi6RE15AnEjtrzu--PUBFuWeqwIcPdNXX0XhvxaYUg
## CODE

In this section, we demonstrate the training process of the repetition transformer (R-Transformer). Please see code demonstration (./code/) for the core part and training process of R-Transformer. Some dependencies are necessary to view and run the code:
  
  • Python 3.6
  
  • Jupyter notebook
  
  • Required packages
    
    – Miditoolkit 
    
    – Pytorch-fast-transformers


## MUSIC DEMONSTRATION

In this section, we demonstrate two music examples: the beauty of repetition in Beethoven’s Fifth Symphony, repetition generation and its application in machine composition. In the first demonstration, we show how Beethoven’s Fifth Symphony is composed based on the “fate motif” and its variants. Please refer to Demonstration of Beethoven’s Fifth Symphony (./music_demonstration/Demo_1/demo1.html). In the second demonstration, we show that the R-Transformer can not only generate diverse and beautiful repetitions, but also achieve pleasant and complex melodies based on one motif. Please refer to Repetition generation and its application in machine composition (./music_demonstration/Demo_2/demo2.html) .


## References
[1] W.-Y. Hsiao, J.-Y. Liu, Y.-C. Yeh, and Y.-H. Yang, “Compound word transformer: Learning to compose full-song music over dynamic directed hypergraphs,” arXiv preprint arXiv:2101.02402 (2021).
