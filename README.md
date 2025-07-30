# AI-Detection-Model
  #### How to Run the Detection Software for Helmets and People:
      This project is meant to be deployed at construction sites to detect people without helmets on

  #### NEEDED SOFTWARE:
      - jetson-inference (https://github.com/dusty-nv/jetson-inference)
      - Python

  #### INSTALLATION
      For this project to run, you need jetson-inference.
      To install jetson-inference, you need to run this in your terminal:

          git clone --recursive https://github.com/dusty-nv/jetson-inference
          mkdir build #this makes the build directory which will store all of your AI models
          cd build
          cmake ../ 
          make -j$(nproc)
          sudo make install
          sudo ldconfig
            
          #just in case pytorch didnt install use this:
          cd ~/jetson-inference/build
          ./install-pytorch.sh

  #### SETUP
      After the install has finished you then need to move this model over to the models directory of the jetson inference detection
      If you are on Linux you can do this using the mv command into the models directory. Then you would need to define your NET variables. You do this by typing this into the terminal:

          NET=~/jetson-inference/.../models/model-name 
          #all this does is define a path to the model and files so route it to wherever you have your models saved in jetson-inference
            


    
  #### RUNNING THE MODEL 
      To run this model properly you need to have installed jetson-inference. After you have installed all of that stuff you can then use detectNet to run this model using this script:

          detectnet \
          --model=$NET/ssd-mobilenet.onnx \
          --labels=$NET/labels.txt \
          --input-blob=input_0 \
          --output-cvg=scores \
          --output-bbox=boxes \

          #if you want live video feed use this command in your root terminal before running detectNet
          camera-capture /dev/video0
          #then use this at the end of the detectNet script:
          /dev/video0

          #if you only want to test images, you will need to set a DATASET variable in the same way you set a NET variable
          DATASET=~/jetson-inference/.../(imagefolder)
          #then you would use this at the end of the detectNet script:
          $DATASET image.jpg output.jpg
