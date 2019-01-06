import os
from subprocess import call

# Select GPU to run:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


call(['python', 'main_MAML.py',
      '--run-name', 'Omniglot2',
      '--data-source', 'Omniglot',
      '--data-transform', 'Rotate90',
      '--N_Way', '5',
      '--K_Shot_MetaTrain', '1',
      '--K_Shot_MetaTest', '1',
      '--n_train_tasks', '0',
      '--model-name', 'OmConvNet',
      # MAML hyper-parameters:
      '--alpha', '0.4',
      '--n_meta_train_grad_steps', '1',
      '--n_meta_train_iterations', '40000', # 40000
      '--meta_batch_size', '32',
      '--n_meta_test_grad_steps', '3',
      '--MAML_Use_Test_Data', 'True',
      ])
