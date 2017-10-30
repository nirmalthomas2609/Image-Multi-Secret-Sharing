from Encrypt import *
import cv2
print ("Please enter the filename of the image to be encrypted")
file_name = input()
img = cv2.imread(file_name)
cv2.imwrite("Original_Image/img.jpg", img)
#a, b, c = cv2.split(img)
print ("The initial shape of the image is {} and the type of the matrix is {}".format(img.shape, type(img)))
print ("Please enter the value of n, t and k seperated by spaces")
n, t, k = list(map(int, input().split(" ")))
print ("Enter 1 to display the histogram else enter 0")
temp = int(input())
plot_hist = True if temp == 1 else False
print ("Enter 1 to show the input image else enter 0")
temp = int(input())
show_image = True if temp == 1 else False

test = Image_Encryption(n=n, t=t, k=k, img=img, show_image=temp, self_debug=True)
#testb = Image_Encryption(n=n, t=t, k=k, img=b, show_image=temp, self_debug=False)
#testc = Image_Encryption(n=n, t=t, k=k, img=c, show_image=temp, self_debug=False)

shadow_imagesa = test.generate_shadow_images(store_shadows=True)
#shadow_imagesb = testb.generate_shadow_images(store_shadows=True)
#shadow_imagesc = testc.generate_shadow_images(store_shadows=True)
print ("Shadow Images stored in folder Shadows")

print ("Please enter the number of shadow images to be used for decryption")
num_available = int(input())
decrypted_image = test.decrypt_image(shadow_imagesa, num_available)
#decrypted_imageb = testb.decrypt_image(shadow_imagesb, num_available)
#decrypted_imagec = testc.decrypt_image(shadow_imagesc, num_available)
print ("Decrypted Image stored in folder Decrypted_Image") 

#t = np.vstack([decrypted_imagea, decrypted_imageb, decrypted_imagec])
print (decrypted_image.shape)
