----------------------------- Explenation --------------------------------

~ Train ~
1. Open one of the scripts in /CNN_Matlab.
2. Change "imageFolder" and "labelFolder" variables.
3. Run the script.

~ Test ~
1. Open "CNN_RAB_XY.m" or "CNN_RAB_Radi.m".
2. Change "imageFolder" and "labelFolder" variables.
3. Copy "NetworkCNN_XY_V2.mat" or "NetworkCNNV10.mat" (corrisponding to the 
   chosen script) into the same directory as the script and rename it to
   NetworkCNN.mat".
4. Run sections: "Parameters and Folders", "Create DataSet", "Load Previous 
   Weights (if possible)", "Test", "Calculate accuracy" and "Visualize".

~ Retrieve sattelite images ~
1. Open the URL: "https://roundabouts.kittelson.com/Roundabouts/Search#".
2. Click "Advanced Search".
3. Specify "Country", "Roundabout Type" and "Status".
4. Click "Search" and "Export as KML".
5. Paste the file in the same folder as "GetRoundaboutPictures.mat".
6. Open and run the script "GetRoundaboutPictures.mat".


----------------------------- Directories --------------------------------

/CNN_Matlab
	/CNN_RAB.m
		Script for training a CNN that outputs X,Y and radius.

	/CNN_RAB_LOOP.m
		Script for training a CNN that outputs X,Y and radius
		with changing initializations.

	/CNN_RAB_Radi.m
		Script for training a CNN that outputs radius.

	/CNN_RAB_XY.m
		Script for training a CNN that outputs X and Y.

/Previous_Weights
	/NetworkCNN_XY_V2.mat
		Contains the weights for the CNN that outputs X and Y.

	/NetworkCNNV10.mat
		Contains the weights for the CNN that outputs radius.

/RealDataSet
	Contains the aerial pictures with a resolution 512x512.

/RealDataSet_128
	Contains the aerial pictures with a resolution 128x128.

/RealDataSet_224
	Contains the aerial pictures with a resolution 224x224.

/GetRoundaboutPictures.m
	Script for retrieving satelite photo's of roundabouts using a database.

/RABInfoRealDataSet.txt
	Contains the labels for the Dataset's.

 
		