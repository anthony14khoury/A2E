# A2E

To run a program that displays a live image captured from the leap motion controller:

    1. Must run on Windows
    2. Download the Leap motion tracking service, Orion v3.2.1:
        https://developer.leapmotion.com/releases/leap-motion-orion-321-39frn-3b659
    3. Plug in Leap Motion controller and ensure Leap Motion Tracking service is running
    4. Download Microsoft Visual Studios 2012 for x86
    5. Install latest version of openCV for windows
    6. Within a new terminal enter the following command:
        "c:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\vcvarsall.bat" x86
    7. From the project root, enter the following commands:
        cd Get_Leap_Image
        cd Sample
        cl /EHsc /I ..\include Sample.cpp /link ..\lib\x86\*.lib /OUT:Release\Sample.exe
        cd Release
        Sample.exe
    8. You should then see a window appear with an image captured by the leap motion controller

To Build the ML Model for sign language prediction:

    1. Open a google colab notebook
    2. Ensure that it is running in python 2.7
    3. Import the train.ipynb script
    4. Upload the Image_Directory.zip
    5. Simply execute every block of code
    6. The model will be generated as a file called "model.h5" in your workspace
    7. Download the model to your local machine and the process will be complete

To perform a live prediction:

    1. From the workspace directory, within the Current Model directory, open the predict.py script
    2. Ensure you have python version 2.7.x
    3. Using pip, install all of the necessary libraries
    4. As you can see it requires an "letter.jpg" file to load in from the same path it is in. So, simply move an image to that path from the Image_Directory dataset or from an image you can capture yourself from the leap motion controller. Ensure that it is called "letter.jpg".
    5. Run the script and observe the prediction and confidence outputted in the terminal.
