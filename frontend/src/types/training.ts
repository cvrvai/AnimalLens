export interface EpochMetric {
  epoch: number;
  total_epochs: number;
  box_loss: number;
  cls_loss: number;
  dfl_loss: number;
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
  timestamp: number;
}

export interface TrainingReport {
  project_dir: string;
  best_weights_path: string;
  last_weights_path: string;
  onnx_weights_path?: string;
  epochs_completed: number;
  map50: number;
  map50_95: number;
  status: string;
}

export interface TrainingJob {
  job_id: string;
  species: string;
  base_model: string;
  dataset_yaml: string;
  epochs: number;
  batch: number;
  imgsz: number;
  device: string;
  experiment_name: string;
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  current_epoch: number;
  progress_pct: number;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  error_message?: string;
  latest_metrics?: EpochMetric;
  history: EpochMetric[];
  report?: TrainingReport;
}

export interface StartTrainingRequest {
  species: string;
  dataset_yaml: string;
  base_model?: string;
  epochs?: number;
  batch?: number;
  imgsz?: number;
  device?: string;
  experiment_name?: string;
}
