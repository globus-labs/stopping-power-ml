import keras
import types
import tempfile

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

localIPP = {
    "sites" : [
        { "site" : "Local_IPP",
          "auth" : {
              "channel" : None,
          },
          "execution" : {
              "executor" : "ipp",
              "provider" : "local",  # LIKELY SHOULD BE BOUND TO SITE
              "block" : { # Definition of a block
                  "taskBlocks" : 3,       # total tasks in a block
                  "initBlocks" : 3,
              }
          }
        }]
}
