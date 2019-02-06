import yolo
import matplotlib.pyplot as plt

def main():
    #First need to load training data
    training, validation = yolo.loadData((224, 224))
    #print(training[0][0])
    #plt.imshow(training[0][0][0])
    #plt.show()

    #create pre-training model
    model = yolo.preTrainModel(224, 224)
    print(model.summary())
    model.compile(optimizer=yolo.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(training, steps_per_epoch=27644//16, epochs=3, validation_data=validation, validation_steps=185)

    #save model
    model.save('preTrainModel.h5')

    return

if __name__ == "__main__":
    main()