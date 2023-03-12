# Signal Approximation by neural network

We will reproduce some experiment of the paper "Implicit Neural Representations with Periodic Activation Functions" (Sitzmann, Martel et al.)
https://www.vincentsitzmann.com/siren/






# We will benchmark different way to extract the data to obtain the pixel prediction

The extraction :


    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_3_1.png)
    





![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_4_2.png)
    



![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_5_2.png)
    


The test of the model :


```python
#Verify all model and show data
model = simple_mlp()
reconstruct_image(model,1,"SimpleMlp Image Reconstruct")

model = fourier_mlp()
reconstruct_image(model,2,"FourierMlp Image Reconstruct")

model = siren_mlp()
reconstruct_image(model,3,"SirenMlp Reconstruct")

model = triangular_mlp()
reconstruct_image(model,4,"TriangularMlp Reconstruct")


```


    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_7_0.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_7_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_7_2.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_7_3.png)
    


The data :


```python
show_image(cv2.imread("lena.png")/255.0,5)
```


    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_9_0.png)
    


# Result of the image prediction


```python
model = simple_mlp()
train_model(model)
reconstruct_image(model,1,"SimpleMlp Image Reconstruct")
print(model)
```

    Epoch: 1, Loss_train: 0.375 
    Epoch: 2, Loss_train: 0.028 
    Epoch: 3, Loss_train: 0.025 
    Epoch: 4, Loss_train: 0.023 
    Epoch: 5, Loss_train: 0.022 
    Epoch: 6, Loss_train: 0.020 
    Epoch: 7, Loss_train: 0.018 
    Epoch: 8, Loss_train: 0.019 
    Epoch: 9, Loss_train: 0.017 
    Epoch: 10, Loss_train: 0.015 
    Finished with a loss for 100 last loss :  0.014891304
    Finished with a loss for 50 last loss :  0.014766498
    simple_mlp(
      (fc_1): Linear(in_features=2, out_features=64, bias=True)
      (fc_2): Linear(in_features=64, out_features=64, bias=True)
      (fc_3): Linear(in_features=64, out_features=3, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_11_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_11_2.png)
    



```python
model = fourier_mlp()
train_model(model)
reconstruct_image(model,2,"FourierMlp Reconstruct")
print(model)
```

    Epoch: 1, Loss_train: 0.281 
    Epoch: 2, Loss_train: 0.009 
    Epoch: 3, Loss_train: 0.008 
    Epoch: 4, Loss_train: 0.006 
    Epoch: 5, Loss_train: 0.005 
    Epoch: 6, Loss_train: 0.005 
    Epoch: 7, Loss_train: 0.005 
    Epoch: 8, Loss_train: 0.004 
    Epoch: 9, Loss_train: 0.004 
    Epoch: 10, Loss_train: 0.004 
    Finished with a loss for 100 last loss :  0.0036816054
    Finished with a loss for 50 last loss :  0.003639353
    fourier_mlp(
      (fourier_1): fourier_extract_full(in_features=2, out_features=81, bias=True)
      (fc_2): Linear(in_features=81, out_features=64, bias=True)
      (fc_3): Linear(in_features=64, out_features=64, bias=True)
      (fc_4): Linear(in_features=64, out_features=3, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_12_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_12_2.png)
    



```python
model = siren_mlp()
train_model(model)
reconstruct_image(model,4,"SirenMlp Reconstruct")
print(model)
```

    Epoch: 1, Loss_train: 0.283 
    Epoch: 2, Loss_train: 0.007 
    Epoch: 3, Loss_train: 0.006 
    Epoch: 4, Loss_train: 0.007 
    Epoch: 5, Loss_train: 0.006 
    Epoch: 6, Loss_train: 0.005 
    Epoch: 7, Loss_train: 0.004 
    Epoch: 8, Loss_train: 0.005 
    Epoch: 9, Loss_train: 0.005 
    Epoch: 10, Loss_train: 0.005 
    Finished with a loss for 100 last loss :  0.0047096624
    Finished with a loss for 50 last loss :  0.00461373
    siren_mlp(
      (siren_r): Siren(
        (net): Sequential(
          (0): SineLayer(
            (linear): Linear(in_features=2, out_features=64, bias=True)
          )
          (1): SineLayer(
            (linear): Linear(in_features=64, out_features=64, bias=True)
          )
          (2): Linear(in_features=64, out_features=1, bias=True)
        )
      )
      (siren_g): Siren(
        (net): Sequential(
          (0): SineLayer(
            (linear): Linear(in_features=2, out_features=64, bias=True)
          )
          (1): SineLayer(
            (linear): Linear(in_features=64, out_features=64, bias=True)
          )
          (2): Linear(in_features=64, out_features=1, bias=True)
        )
      )
      (siren_b): Siren(
        (net): Sequential(
          (0): SineLayer(
            (linear): Linear(in_features=2, out_features=64, bias=True)
          )
          (1): SineLayer(
            (linear): Linear(in_features=64, out_features=64, bias=True)
          )
          (2): Linear(in_features=64, out_features=1, bias=True)
        )
      )
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_13_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_13_2.png)
    



```python
model = triangular_mlp()
train_model(model)
reconstruct_image(model,4,"TriangularMlp Reconstruct")
print(model)

```

    Epoch: 1, Loss_train: 0.304 
    Epoch: 2, Loss_train: 0.020 
    Epoch: 3, Loss_train: 0.018 
    Epoch: 4, Loss_train: 0.016 
    Epoch: 5, Loss_train: 0.014 
    Epoch: 6, Loss_train: 0.014 
    Epoch: 7, Loss_train: 0.014 
    Epoch: 8, Loss_train: 0.012 
    Epoch: 9, Loss_train: 0.012 
    Epoch: 10, Loss_train: 0.013 
    Finished with a loss for 100 last loss :  0.011705444
    Finished with a loss for 50 last loss :  0.011567899
    triangular_mlp(
      (triangular_extraction): triangular_features_extraction(in_features=2, out_features=16, bias=True)
      (fc_1): Linear(in_features=16, out_features=64, bias=True)
      (fc_2): Linear(in_features=64, out_features=3, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_14_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_14_2.png)
    


# After the image reconstruction, the 2D signal reconstruction :

### 



The 2D signal to Reconstruct :




    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_19_0.png)
    


To solve this task, we can choose between 2 types of neural network :

    minimal : Obtain reconstruction with the smaller model to see the benefits and limits of each way to extract the data
    maximal : Obtain a reconstruction with the higher accuracy (with bigger model), to test the model on harder task after that




```python
#Verify models and data 
model = simple_mlp_Helmholtz()
reconstruct_Helmholtz(model,index=1)

model = fourier_mlp_Helmholtz()
reconstruct_Helmholtz(model,index=2)

model = siren_mlp_Helmholtz()
reconstruct_Helmholtz(model,index=3)

model = triangular_mlp_Helmholtz()
reconstruct_Helmholtz(model,index=4)


(data,label) = create_dataset_Helmholtz(size=128)
show_data(data,label,5,size=128)

```


    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_23_0.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_23_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_23_2.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_23_3.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_23_4.png)
    


# Result of the 2D signal approximation, easy version


```python
#Simple data
(data,label) = create_dataset_Helmholtz(size=128)
show_data(data,label,5,size=128)
```


    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_25_0.png)
    



```python
model = simple_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=1,title="SimpleMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.490 
    Epoch: 51, Loss_train: 0.373 
    Epoch: 101, Loss_train: 0.232 
    Epoch: 151, Loss_train: 0.107 
    Epoch: 201, Loss_train: 0.065 
    Epoch: 251, Loss_train: 0.042 
    Epoch: 301, Loss_train: 0.034 
    Epoch: 351, Loss_train: 0.030 
    Epoch: 401, Loss_train: 0.026 
    Epoch: 451, Loss_train: 0.023 
    Finished with a loss for 100 last loss :  0.023761917
    Finished with a loss for 50 last loss :  0.023563698
    simple_mlp_Helmholtz(
      (fc_1): Linear(in_features=2, out_features=32, bias=True)
      (fc_2): Linear(in_features=32, out_features=32, bias=True)
      (fc_3): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_26_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_26_2.png)
    



```python
model = siren_fake_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=1,title="SirenFakeMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.527 
    Epoch: 51, Loss_train: 0.497 
    Epoch: 101, Loss_train: 0.475 
    Epoch: 151, Loss_train: 0.478 
    Epoch: 201, Loss_train: 0.498 
    Epoch: 251, Loss_train: 0.452 
    Epoch: 301, Loss_train: 0.479 
    Epoch: 351, Loss_train: 0.460 
    Epoch: 401, Loss_train: 0.432 
    Epoch: 451, Loss_train: 0.370 
    Finished with a loss for 100 last loss :  0.32091698
    Finished with a loss for 50 last loss :  0.31662843
    siren_fake_mlp_Helmholtz(
      (fc_1): Linear(in_features=2, out_features=32, bias=True)
      (fc_2): Linear(in_features=32, out_features=32, bias=True)
      (fc_3): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_27_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_27_2.png)
    



```python

model = fourier_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=2,title="FourierMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.509 
    Epoch: 51, Loss_train: 0.001 
    Epoch: 101, Loss_train: 0.001 
    Epoch: 151, Loss_train: 0.000 
    Epoch: 201, Loss_train: 0.000 
    Epoch: 251, Loss_train: 0.000 
    Epoch: 301, Loss_train: 0.000 
    Epoch: 351, Loss_train: 0.000 
    Epoch: 401, Loss_train: 0.000 
    Epoch: 451, Loss_train: 0.000 
    Finished with a loss for 100 last loss :  0.00023157937
    Finished with a loss for 50 last loss :  0.00023680594
    fourier_mlp_Helmholtz(
      (fourier_1): fourier_extract_full(in_features=2, out_features=81, bias=True)
      (fc_2): Linear(in_features=81, out_features=32, bias=True)
      (fc_3): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_28_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_28_2.png)
    



```python
model = siren_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=4,title="SirenMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 1.144 
    Epoch: 51, Loss_train: 0.002 
    Epoch: 101, Loss_train: 0.001 
    Epoch: 151, Loss_train: 0.002 
    Epoch: 201, Loss_train: 0.003 
    Epoch: 251, Loss_train: 0.001 
    Epoch: 301, Loss_train: 0.001 
    Epoch: 351, Loss_train: 0.001 
    Epoch: 401, Loss_train: 0.001 
    Epoch: 451, Loss_train: 0.001 
    Finished with a loss for 100 last loss :  0.0010300127
    Finished with a loss for 50 last loss :  0.0010263402
    siren_mlp_Helmholtz(
      (siren): Siren(
        (net): Sequential(
          (0): SineLayer(
            (linear): Linear(in_features=2, out_features=32, bias=True)
          )
          (1): SineLayer(
            (linear): Linear(in_features=32, out_features=32, bias=True)
          )
          (2): SineLayer(
            (linear): Linear(in_features=32, out_features=1, bias=True)
          )
        )
      )
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_29_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_29_2.png)
    



```python
model = triangular_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=5,title="TriangularMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.502 
    Epoch: 51, Loss_train: 0.109 
    Epoch: 101, Loss_train: 0.021 
    Epoch: 151, Loss_train: 0.011 
    Epoch: 201, Loss_train: 0.008 
    Epoch: 251, Loss_train: 0.008 
    Epoch: 301, Loss_train: 0.006 
    Epoch: 351, Loss_train: 0.006 
    Epoch: 401, Loss_train: 0.005 
    Epoch: 451, Loss_train: 0.005 
    Finished with a loss for 100 last loss :  0.0049856286
    Finished with a loss for 50 last loss :  0.004961751
    triangular_mlp_Helmholtz(
      (input_decompo): triangular_features_extraction(in_features=2, out_features=16, bias=True)
      (fc_1): Linear(in_features=16, out_features=32, bias=True)
      (fc_2): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_30_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_30_2.png)
    


# Result of the 2D signal approximation,  moderate version


```python
#moderate data
(data,label) = create_dataset_Helmholtz_max(size=128)
show_data(data,label,5,size=128)
```


    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_32_0.png)
    



```python
model = simple_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz_max(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=1,title="SimpleMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.565 
    Epoch: 51, Loss_train: 0.532 
    Epoch: 101, Loss_train: 0.494 
    Epoch: 151, Loss_train: 0.498 
    Epoch: 201, Loss_train: 0.496 
    Epoch: 251, Loss_train: 0.490 
    Epoch: 301, Loss_train: 0.501 
    Epoch: 351, Loss_train: 0.490 
    Epoch: 401, Loss_train: 0.499 
    Epoch: 451, Loss_train: 0.475 
    Finished with a loss for 100 last loss :  0.48520267
    Finished with a loss for 50 last loss :  0.4854953
    simple_mlp_Helmholtz(
      (fc_1): Linear(in_features=2, out_features=32, bias=True)
      (fc_2): Linear(in_features=32, out_features=32, bias=True)
      (fc_3): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_33_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_33_2.png)
    



```python
model = fourier_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz_max(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=2,title="FourierMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.515 
    Epoch: 51, Loss_train: 0.016 
    Epoch: 101, Loss_train: 0.006 
    Epoch: 151, Loss_train: 0.006 
    Epoch: 201, Loss_train: 0.004 
    Epoch: 251, Loss_train: 0.004 
    Epoch: 301, Loss_train: 0.004 
    Epoch: 351, Loss_train: 0.003 
    Epoch: 401, Loss_train: 0.003 
    Epoch: 451, Loss_train: 0.003 
    Finished with a loss for 100 last loss :  0.0035549884
    Finished with a loss for 50 last loss :  0.0035740403
    fourier_mlp_Helmholtz(
      (fourier_1): fourier_extract_full(in_features=2, out_features=81, bias=True)
      (fc_2): Linear(in_features=81, out_features=32, bias=True)
      (fc_3): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_34_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_34_2.png)
    



```python
model = siren_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz_max(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=4,title="SirenMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.986 
    Epoch: 51, Loss_train: 0.029 
    Epoch: 101, Loss_train: 0.019 
    Epoch: 151, Loss_train: 0.019 
    Epoch: 201, Loss_train: 0.015 
    Epoch: 251, Loss_train: 0.022 
    Epoch: 301, Loss_train: 0.019 
    Epoch: 351, Loss_train: 0.022 
    Epoch: 401, Loss_train: 0.015 
    Epoch: 451, Loss_train: 0.027 
    Finished with a loss for 100 last loss :  0.020570038
    Finished with a loss for 50 last loss :  0.020684015
    siren_mlp_Helmholtz(
      (siren): Siren(
        (net): Sequential(
          (0): SineLayer(
            (linear): Linear(in_features=2, out_features=32, bias=True)
          )
          (1): SineLayer(
            (linear): Linear(in_features=32, out_features=32, bias=True)
          )
          (2): SineLayer(
            (linear): Linear(in_features=32, out_features=1, bias=True)
          )
        )
      )
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_35_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_35_2.png)
    



```python
model = triangular_mlp_Helmholtz()
train_model(model,create_dataset_Helmholtz_max(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=5,title="TriangularMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.530 
    Epoch: 51, Loss_train: 0.502 
    Epoch: 101, Loss_train: 0.497 
    Epoch: 151, Loss_train: 0.476 
    Epoch: 201, Loss_train: 0.444 
    Epoch: 251, Loss_train: 0.465 
    Epoch: 301, Loss_train: 0.439 
    Epoch: 351, Loss_train: 0.417 
    Epoch: 401, Loss_train: 0.417 
    Epoch: 451, Loss_train: 0.409 
    Finished with a loss for 100 last loss :  0.39978424
    Finished with a loss for 50 last loss :  0.40015367
    triangular_mlp_Helmholtz(
      (input_decompo): triangular_features_extraction(in_features=2, out_features=16, bias=True)
      (fc_1): Linear(in_features=16, out_features=32, bias=True)
      (fc_2): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_36_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_36_2.png)
    



```python

```

# Result of the 2D signal approximation,  hard version


```python
#hard data
(data,label) = create_dataset_hard(size=128)
show_data(data,label,5,size=128)
```


    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_39_0.png)
    



```python
model = simple_mlp_Helmholtz()
train_model(model,create_dataset_hard(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=1,title="SimpleMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.508 
    Epoch: 51, Loss_train: 0.313 
    Epoch: 101, Loss_train: 0.262 
    Epoch: 151, Loss_train: 0.250 
    Epoch: 201, Loss_train: 0.203 
    Epoch: 251, Loss_train: 0.203 
    Epoch: 301, Loss_train: 0.177 
    Epoch: 351, Loss_train: 0.188 
    Epoch: 401, Loss_train: 0.166 
    Epoch: 451, Loss_train: 0.165 
    Finished with a loss for 100 last loss :  0.16602142
    Finished with a loss for 50 last loss :  0.16555744
    simple_mlp_Helmholtz(
      (fc_1): Linear(in_features=2, out_features=32, bias=True)
      (fc_2): Linear(in_features=32, out_features=32, bias=True)
      (fc_3): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_40_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_40_2.png)
    



```python
model = fourier_mlp_Helmholtz()
train_model(model,create_dataset_hard(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=2,title="FourierMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.519 
    Epoch: 51, Loss_train: 0.043 
    Epoch: 101, Loss_train: 0.045 
    Epoch: 151, Loss_train: 0.037 
    Epoch: 201, Loss_train: 0.041 
    Epoch: 251, Loss_train: 0.040 
    Epoch: 301, Loss_train: 0.038 
    Epoch: 351, Loss_train: 0.037 
    Epoch: 401, Loss_train: 0.042 
    Epoch: 451, Loss_train: 0.036 
    Finished with a loss for 100 last loss :  0.03886108
    Finished with a loss for 50 last loss :  0.038700297
    fourier_mlp_Helmholtz(
      (fourier_1): fourier_extract_full(in_features=2, out_features=81, bias=True)
      (fc_2): Linear(in_features=81, out_features=32, bias=True)
      (fc_3): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_41_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_41_2.png)
    



```python
model = siren_mlp_Helmholtz()
train_model(model,create_dataset_hard(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=4,title="SirenMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.938 
    Epoch: 51, Loss_train: 0.044 
    Epoch: 101, Loss_train: 0.042 
    Epoch: 151, Loss_train: 0.043 
    Epoch: 201, Loss_train: 0.040 
    Epoch: 251, Loss_train: 0.040 
    Epoch: 301, Loss_train: 0.040 
    Epoch: 351, Loss_train: 0.045 
    Epoch: 401, Loss_train: 0.044 
    Epoch: 451, Loss_train: 0.041 
    Finished with a loss for 100 last loss :  0.044641916
    Finished with a loss for 50 last loss :  0.044657227
    siren_mlp_Helmholtz(
      (siren): Siren(
        (net): Sequential(
          (0): SineLayer(
            (linear): Linear(in_features=2, out_features=32, bias=True)
          )
          (1): SineLayer(
            (linear): Linear(in_features=32, out_features=32, bias=True)
          )
          (2): SineLayer(
            (linear): Linear(in_features=32, out_features=1, bias=True)
          )
        )
      )
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_42_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_42_2.png)
    



```python
model = triangular_mlp_Helmholtz()
train_model(model,create_dataset_hard(),epochs=500,plot_step=50)
reconstruct_Helmholtz(model,index=5,title="TriangularMLP Sinus Reconstruction")
print(model)
```

    Epoch: 1, Loss_train: 0.496 
    Epoch: 51, Loss_train: 0.334 
    Epoch: 101, Loss_train: 0.277 
    Epoch: 151, Loss_train: 0.258 
    Epoch: 201, Loss_train: 0.226 
    Epoch: 251, Loss_train: 0.214 
    Epoch: 301, Loss_train: 0.196 
    Epoch: 351, Loss_train: 0.193 
    Epoch: 401, Loss_train: 0.185 
    Epoch: 451, Loss_train: 0.185 
    Finished with a loss for 100 last loss :  0.17553586
    Finished with a loss for 50 last loss :  0.17520744
    triangular_mlp_Helmholtz(
      (input_decompo): triangular_features_extraction(in_features=2, out_features=16, bias=True)
      (fc_1): Linear(in_features=16, out_features=32, bias=True)
      (fc_2): Linear(in_features=32, out_features=1, bias=True)
    )



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_43_1.png)
    



    
![png](main_image_reconstruction_CHUPIN_files/main_image_reconstruction_CHUPIN_43_2.png)
    


