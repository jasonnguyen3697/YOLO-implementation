import yolo

def main():
    #First need to load training data
    training, validation = yolo.loadData((224, 224))

    #create pre-training model
    model = yolo.preTrainModel(224, 224)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(training, steps_per_epoch=1728, epochs=3, validation_data=validation, validation_steps=186)

    #save model
    model.save('preTrainModel.h5')

    return

if __name__ == "__main__":
    main()