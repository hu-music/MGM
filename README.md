# The Beauty of Repetition: an Algorithmic Composition Model with Motif-level Repetition Generator and Outline-to-music Generator
This document is designed to assist with reading supplementary materials of the paper ''The Beauty of Repetition: an Algorithmic Composition Model with Motif-level Repetition Generator and Outline-to-music Generator''. In this document, we describe a dataset named Music Repetition Dataset (MRD) and the code of a novel repetition generation system that includes motif-level repetition generator (MRG) and outline-to-music generator (O2MG). Moreover, we demonstrate that the proposed model can generate very interesting results on pop music and fate motif from the classical music.

Demo:  https://zhejinghu.com/mgm/

## DATASET

In this section, we describe basic information of Music Repetition Dataset (MRD) in detail. This dataset contains two parts. The first part is motif-level repetitions, which are extracted based on 1,748 complete pop piano songs from [1]. All songs are in 4/4 time signature, and each song is converted into a symbolic sequence following transcription, synchronization, quantization, and analysis. This part contains 562,563 training data and 21,766 test data with the labels of five motif level repetitions. The second part is outline-music pairs, which are extracted based on pop piano songs from [1] and [2]. Please refer to https://connectpolyu-my.sharepoint.com/:f:/g/personal/19045203r_connect_polyu_hk/EiZ6wi6RE15AnEjtrzu--PUBFuWeqwIcPdNXX0XhvxaYUg for the dataset information. 
## CODE

In this section, we demonstrate the training process of the MRG and O2MG. Please see code demonstration (./code/) for the core part and training process. Some dependencies are necessary to view and run the code:
  
  • Python 3.6
  
  • Jupyter notebook
  
  • Required packages
    
    – Miditoolkit 
    
    – Pytorch-fast-transformers


## MUSIC DEMONSTRATION

In this section, we demonstrate examples in the paper: motif-level examples, end-to-end generation and fine-grained control. Please refer to https://zhejinghu.com/mgm/ for audio samples. In the first demonstration, we first show how Beethoven’s Fifth Symphony is composed based on the "fate motif" and its variants. We then show that MRG can not only generate diverse and beautiful repetitions, but also achieve pleasant and complex melodies based on one motif. In the second demonstration, we show that the proposed model can generate a complete outline and music based on one motif. In the third demonstration, we show that the user can manipulate the outline and the model can generate the music based on the given outline.


## References
[1] W.-Y. Hsiao, J.-Y. Liu, Y.-C. Yeh, and Y.-H. Yang, “Compound word transformer: Learning to compose full-song music over dynamic directed hypergraphs,” arXiv preprint arXiv:2101.02402 (2021).

[2] Z. Wang, K. Chen, J. Jiang, Y. Zhang, M. Xu, S. Dai, X. Gu, and G. Xia, “Pop909: A pop-song dataset for music arrangement generation,” arXiv preprint arXiv:2008.07142 (2020).
