# 📊 Misviz and Misviz-synth datasets 

The Misviz and Misviz-synth datasets are made available under a **CC-BY-SA-4.0** license.

### Misviz-synth

Misviz-synth contains 57,665 synthetic visualizations generated with Matplotlib. It is split into a train, validation, and test set.

- *data/misviz_synth/misviz_synth.json* contains the task labels and metadata
- The visualizations, the underlying data tables, the code, and the axis metadata can be downloaded from [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4782)

Each record contains the following items: 

- `image_path`: local path to the image
- `chart_type`: a list of chart types present in the visualization
- `misleader`: a list of misleaders present in the visualization. If it is empty, the visualization is not misleading.
- `table_id`: the unique identifier of the visualization's underlying data table
- `variant`: the variant for a given (table, misleader, chart type) triplet
- `table_data_path`: local path to the file containing the underlying data table
- `axis_data_path`: local path to the file containing the axis metadata
- `code_path`: local path to the file containing the python code used to draw the visualization
- `split`: the dataset split (train, train small, dev, val, or test)

### Misviz 

Misviz contains 2,604 real-world visualizations collected from various websites. It is split into a dev, validation, and test set.

- *data/misviz/misviz.json* contains the task labels and metadata
- The visualizations can be downloaded from the web using the following script

```python
$ python data/download_misviz_images.py --use_wayback 0
```

```use_wayback``` is a paramater to decide whether the image is scraped from the original URL (0) or from an archived version of the URL on the Wayback Machine (1).
The archived URL serves as a backup.

Please contact  `jonathan.tonglet@tu-darmstadt.de` if you face issues downloading the images of Misviz.

Each record contains the following items: 

- `image_path`: local path to the image
- `image_url`: URL of the image
- `chart_type`: a list of chart types present in the visualization
- `misleader`: a list of misleaders present in the visualization. If it is empty, the visualization is not misleading.
- `wayback_image_url`: URL of the archived image on the Wayback Machine
- `split`: the dataset split (dev, val, or test)
- `bbox`: the list of coordinates of bounding boxes indicating misleading regions of the chart. Empty for most instances.


