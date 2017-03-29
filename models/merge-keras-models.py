from keras.models import load_model, model_from_json


def model_to_keras_v1(model_file, weights_file, output_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)
    loss = 'categorical_crossentropy' if '-mc-' in model_file else 'binary_crossentropy'
    model.compile(loss=loss, optimizer='RMSprop')
    model.save(output_file)


if __name__ == '__main__':
    import os
    classifications = ['ekman', 'plutchik', 'poms']
    for folder in classifications:
        for file in os.listdir(folder):
            if file.endswith('.json'):
                setting = 'mc' if '_mc_' in file else 'ml'
                file = os.path.join(folder, file)
                model_to_keras_v1(
                    file,
                    file.replace('.json', '.h5'),
                    '{}-{}.h5'.format(folder, setting),
                )
    for file in os.listdir('unison'):
        emo = next(c for c in classifications if c in file)
        setting = 'mc' if '_mc_' in file else 'ml'
        m = load_model('unison/'+file)
        m.save('unison-{}-{}.h5'.format(emo, setting))

