from gpiozero import Button
import os
import threading

stopButton = Button(26)  # defines the button as an object and chooses GPIO 26

def check_button():
    while True:  # infinite loop
        if stopButton.is_pressed:  # Check to see if button is pressed
            threading.Event().wait(1)  # wait for the hold time we want
            if stopButton.is_pressed:  # check if the user let go of the button
                os.system("shutdown now -h")  # shut down the Pi -h is or -r will reset
        threading.Event().wait(1)  # wait to loop again so we don?t use the processor too much

# Start the button checking in a separate thread
button_thread = threading.Thread(target=check_button)
button_thread.daemon = True
button_thread.start()

# The main thread can perform other tasks or just sleep
try:
    while True:
        threading.Event().wait(1)
except KeyboardInterrupt:
    print("Program terminated.")
