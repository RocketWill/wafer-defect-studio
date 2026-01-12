# Reference TIAToolbox without making it an MVP runtime dependency

The MVP will reuse the design lessons of TIAToolbox PatchPredictor but will neither fork TIAToolbox nor depend on it at runtime. The expected 5–20 MP grayscale images need only a small native-resolution windowing, batch prediction, CAM, and overlap-aggregation pipeline; a pathology-focused WSI stack would add more integration surface than this use case currently requires.
