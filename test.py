plt.figure(figsize=(256, 256))

ax = plt.subplot(10, 10, 1)
plt.imshow(downsized_images[image_index])

ax = plt.subplot(10, 10, 2)
plt.imshow(downsized_images[image_index], interpolation="bicubic")

# ax = plt.subplot(10, 10, i)
# plt.imshow(encoded_imgs[image_index].reshape((64*64, 256)))

ax = plt.subplot(10, 10, 3)
plt.imshow(sr1[image_index])

ax = plt.subplot(10, 10, 3)
plt.imshow(sr1[image_index])

ax = plt.subplot(10, 10, 4)
plt.imshow(real_images[image_index])

plt.savefig('./')
