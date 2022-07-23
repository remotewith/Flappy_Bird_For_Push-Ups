import pygame
import cv2
import numpy as np
import random   
import thesius_module as the
import mediapipe as mp
import time

#Pose detector
mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils

#"the" module
detector=the.handDetector(maxHands=1)


#initialize
pygame.init()

#create window/display
width,height=1280,720
window=pygame.display.set_mode((width,height))
pygame.display.set_caption("RIVAL SONS")

#Display shit
pygame.font.init() 
my_font = pygame.font.SysFont('Comic Sans MS', 30)


#initialize clock for fps
fps=30
clock=pygame.time.Clock()

#Lives and Score
life=1
score=0

#Load Images
#imgBackground=pygame.image.load('background-black.png').convert()
imgBird=pygame.image.load("flappy1.png").convert_alpha()
imgPipeUp=pygame.image.load("pipe1.png").convert_alpha()
imgPipeDown=pygame.image.load("pipe2.png").convert_alpha()


rectBird=imgBird.get_rect()#this function creates a rectangle around the desired image
rectPipeUp=imgPipeUp.get_rect()
rectPipeDown=imgPipeDown.get_rect()
rectBird.x,rectBird.y=-90,-90
rectPipeUp.x,rectPipeUp.y=1280,500
rectPipeDown.x,rectPipeDown.y=1280,-200
#imgEnemy=pygame.transform.rotate(imgEnemy,90)
#imgEnemy=pygame.transform.flip(imgEnemy,False,True)


cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#variables
speed=50


def resetPipe():
    global speed
    rectPipeDown.y=random.randint(-500,0)
    rectPipeUp.y=rectPipeDown.y+700
    rectPipeUp.x,rectPipeDown.x=1280,1280
    speed+=2

#main loop

start=True
lost=False

while start:
    lmCist=[]

    for event in pygame.event.get():#this function gets all the events frm pygame
        if event.type==pygame.QUIT:
            start=False
            pygame.quit()

    #Apply logic(main thing or content is written under this)
    
    _,img=cap.read()
    #img=cv2.flip(img,1)
    img=detector.findHands(img)
    
    lmList=detector.findPosition(img)

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results=pose.process(imgRGB)

    rectPipeDown.x,rectPipeUp.x=rectPipeDown.x-speed,rectPipeUp.x-speed

    if rectPipeUp.x<0:
        score+=1
        resetPipe()


    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):

            
            if results:

                h,w,c=img.shape            
                x,y=int(lm.x*w),int(lm.y*h)
                
                lmCist.append([id,x,y])
                if id==11:
                    #x,y=lmCist[12][1],lmList[12][2]
                    x=np.interp(x,[0,1240],[1280,0])
                    x=int(x)
                    y=np.interp(y,[0,720],[-20,640])
                    y=int(y)
                    rectBird.x=x
                    rectBird.y=y
                    if rectPipeUp.collidepoint(x,y-10) or rectPipeDown.collidepoint(x,y+20):
                        start=False
    
    #Display
    
    text_surface1 = my_font.render("SCORE:"+str(int(score)), False, (0, 0, 0))


    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgRGB=np.rot90(imgRGB)
    frame=pygame.surfarray.make_surface(imgRGB).convert()
    window.blit(frame,(0,0))
    window.blit(imgPipeUp,rectPipeUp)
    window.blit(imgPipeDown,rectPipeDown)
    window.blit(imgBird,rectBird)
    window.blit(text_surface1, (1140,27))

    #Update Display
    pygame.display.update()

    #set FPS
    clock.tick(fps)
