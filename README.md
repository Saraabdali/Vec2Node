# Vec2Node: Self-training with Tensor Augmentation for Text Classification with Few Labels 
To appear at ECML PKDD 2022.

Authors:Sara Abdali, Subhabrata Mukherjee, Evangelos Papalexakis
#########################################################################################################################
Recent advances in state-of-the-art machine learning models like deep
neural networks heavily rely on large amounts of labeled training data which is
difficult to obtain for many applications. To address label scarcity, recent work
has focused on data augmentation techniques to create synthetic training data. In
this work, we propose a novel approach of data augmentation leveraging tensor
decomposition to generate synthetic samples by exploiting local and global infor-
mation in text and reducing concept drift. We develop Vec2Node that leverages
self-training from in-domain unlabeled data augmented with tensorized word em-
beddings that significantly improves over state-of-the-art models, particularly
in low-resource settings. For instance, with only 1% of labeled training data,
Vec2Node improves the accuracy of a base model by 16.7%. Furthermore,
Vec2Node generates explicable augmented data leveraging tensor embeddings
