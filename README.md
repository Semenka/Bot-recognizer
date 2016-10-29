Computer vision capstone project consists of the following files:
1) index.py: main executable file.
2) telepot folder which includes telepot framework for bot's in Telegram conctruction
3) Report.gdoc main report regarding the whole project

In order to run the project the following steps are needed:
1) Install Telegram messager
2) Create in the telegram chat with bot (found bot): Recognizer
3) Run executable index.py file. Executable file represents functionality for the bot Recognizer.
4) Regonzizer includes the following functionality:
	a) If you send to image in .jpg format, then bot will compare face on the image with faces of famous people in the database.
	b) Recognizer will propose the name to whom proposed by user face is similar.  
	b) Then the keyboard yes/no appeared.
	c) If the description is correct, user could press yes and try next image.
	d) If description is not accurate, user could press no and provide accurate name of the person's face on the image in the format Name_Surname. In this case the image will be added to the database. By default the database location: /Users/SEM/scikit_learn_data/lfw_home/lfw_funneled/. In order to change it, please edit reference in line 48 index.py source code.
5) Power user part. 2 functions have been created for the dataset analsysis:
	a)Bot_image_recognized(image). Used in the project by default. Function used by Recognizer bot for comparison of received image with database of faces of famous people.
	b)face_recognition_test(). Function to analyze database of famouse people and use different options of machine learnint on this dataset (not used bu bot)


