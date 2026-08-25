export interface BoundingBoxYOLO {
  class_id: number;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
}

export interface Keyframe {
  image_name: string;
  split: 'train' | 'val';
  image_url: string;
  bboxes: BoundingBoxYOLO[];
}

export interface Dataset {
  id: string;
  name: string;
  path: string;
  dataset_yaml: string;
  total_images: number;
  train_images: number;
  val_images: number;
  classes: Record<number, string>;
  preview_thumbnail?: string;
}

export interface DatasetUploadResponse {
  status: string;
  dataset_name: string;
  species: string;
  extracted_frames: number;
  train_frames: number;
  val_frames: number;
  pseudo_labels_generated: number;
  dataset_yaml: string;
}
