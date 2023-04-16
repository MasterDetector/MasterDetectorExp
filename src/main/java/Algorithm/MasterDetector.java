package Algorithm;


import Algorithm.util.KDTreeUtil;
import Algorithm.util.VARUtil;

import java.util.ArrayList;

public class MasterDetector {
    private final double[][] td;
    private double[][] td_repaired;
    private boolean[] td_anomalies;
    //    private boolean[] anomalies_in_repaired;
//    private double[][] td_prediction;
    private final KDTreeUtil kdTreeUtil;
    private final long[] td_time;
    private final int columnCnt;
    private final int k;
    private final int p;
    private double[] std;
    private int[] initial_window;

    private VARUtil prediction_model;

    private final int n;

    private double eta;
    private final long cost_time;

    public MasterDetector(double[][] td, KDTreeUtil kdTreeUtil, long[] td_time, int columnCnt, int k, int p, double eta) {
        this.td = td;
        this.kdTreeUtil = kdTreeUtil;
        this.td_time = td_time;
        this.columnCnt = columnCnt;
        this.k = k;
        this.p = p;
        this.eta = eta;
        this.n = td.length;
        long startTime = System.currentTimeMillis();
        this.repair();
        long endTime = System.currentTimeMillis();
        this.cost_time = endTime - startTime;
        System.out.println("MasterRepair time cost:" + cost_time + "ms");
    }

    public double delta(double[] t_tuple, double[] m_tuple) {
        double distance = 0d;
        for (int pos = 0; pos < columnCnt; pos++) {
            double temp = t_tuple[pos] - m_tuple[pos];
            temp = temp / std[pos];
            distance += temp * temp;
        }
        distance = Math.sqrt(distance);
        return distance;
    }

    private double varianceImperative(double[] value) {
        double average = 0.0;
        int cnt = 0;
        for (double p : value) {
            if (!Double.isNaN(p)) {
                cnt += 1;
                average += p;
            }
        }
        if (cnt == 0) {
            return 0d;
        }
        average /= cnt;

        double variance = 0.0;
        for (double p : value) {
            if (!Double.isNaN(p)) {
                variance += (p - average) * (p - average);
            }
        }
        return variance / cnt;
    }

    private double[] getColumn(int pos) {
        double[] column = new double[n];
        for (int i = 0; i < n; i++) {
            column[i] = this.td[i][pos];
        }
        return column;
    }

    public void call_std() {
        this.std = new double[this.columnCnt];
        for (int i = 0; i < this.columnCnt; i++) {
            std[i] = Math.sqrt(varianceImperative(getColumn(i)));
        }
    }

    public boolean checkConsistency(double[] tuple) {
        double[] NN = kdTreeUtil.nearestNeighbor(tuple);
        double delta = delta(tuple, NN);
        if (delta > eta) {
            return false;
        } else return true;
    }

    public void getOriginalAnomaliesAndLearnModel() {
//        int left = 0;
//        int right = 0;
//        int m = 10;
        ArrayList<ArrayList<Double>> learning_samples = new ArrayList<>();
//        int samples_cnt = 0;
        td_anomalies = new boolean[n];
        for (int i = 0; i < td.length; i++) {
            double[] tuple = td[i];
            boolean isNormal = checkConsistency(tuple);
            td_anomalies[i] = !isNormal;

//            if (samples_cnt < m) {
//                if (right - left + 1 == p) {
//                    samples_cnt++;
//                    for (int j = left; j <= right; j++) {
//                        ArrayList<Double> sample = new ArrayList<>();
//                        for (double value : td[j]) {
//                            sample.add(value);
//                        }
//                        learning_samples.add(sample);
//                    }
//                }
//            }
            if (isNormal) {
                ArrayList<Double> sample = new ArrayList<>();
                for (double value : tuple) {
                    sample.add(value);
                }
                learning_samples.add(sample);
//                right++;
            }
//            else {
//                left = i + 1;
//                right = i + 1;
//            }
        }
        this.prediction_model = new VARUtil(columnCnt);
        this.prediction_model.fit(learning_samples);
    }

    //
    public void findInitialWindow(int p) {
        initial_window = new int[2];
        int left = 0;
        int right = 0;
        for (int i = 0; i < td_anomalies.length; i++) {
            if (right - left + 1 == p) {
                initial_window[0] = left;
                initial_window[1] = right;
                break;
            }
            if (td_anomalies[i] == Boolean.TRUE) {
                left = i + 1;
                right = i + 1;
            } else {
                right++;
            }
        }
    }

    public double[][] getWindow(double[][] data, int i, int p) {
        if (i < p) {
            System.out.println("ERROR: i must be greater than p.");
            return null;
        }
        double[][] W = new double[p][columnCnt];
        System.arraycopy(data, i - p, W, 0, p);
        return W;
    }

    //    //  public double[] findTheOptimalForwardRepair(
//    //      double[][] candidates, int i) {
//    //    double[][] W_repaired = getWindow(this.td_repaired, i, p);
//    //    double[] x_repaired_predicted = prediction_model.predict(W_repaired);
//    //    double[] optimal_repair = new ArrayList<>();
//    //    double min_dis = Double.MAX_VALUE;
//    //    for (double[] candidate : candidates) {
//    //      if (delta(x_repaired_predicted, candidate) < min_dis) {
//    //        min_dis = delta(x_repaired_predicted, candidate);
//    //        optimal_repair = candidate;
//    //      }
//    //    }
//    //    return optimal_repair;
//    //  }
//
//    //  public double[] findTheOptimalBackwardRepair(
//    //      double[][] candidates, int i) {
//    //    double[][] W_repaired = getWindow(this.td_repaired, i, p);
//    //    double[] x_repaired = this.td_repaired.get(i);
//    //    double[] optimal_repair = new ArrayList<>();
//    //    double min_dis = Double.MAX_VALUE;
//    //    for (double[] candidate : candidates) {
//    //      W_repaired.set(0, candidate);
//    //      double[] x_repaired_predicted = prediction_model.predict(W_repaired);
//    //
//    //      if (delta(x_repaired_predicted, x_repaired) < min_dis) {
//    //        min_dis = delta(x_repaired_predicted, x_repaired);
//    //        optimal_repair = candidate;
//    //      }
//    //    }
//    //    return optimal_repair;
//    //  }
//
    public void forwardRepairing(int p) {
        int i = initial_window[1] + 1;

        while (i < n) {
            if (td_anomalies[i] == Boolean.TRUE) {
                double[][] W_repaired = getWindow(this.td_repaired, i, p);
                double[] x_repaired_predicted = arrayToList(prediction_model.predict(W_repaired));
//                double[][] W_prediction = getWindow(this.td_prediction, i, p);
//                double[] x_predicted = arrayToList(prediction_model.predict(W_prediction));
                double[][] candidates =
                        this.kdTreeUtil.kNearestNeighbors(x_repaired_predicted, this.k);
                //        find the optimal repair
                double[] optimal_repair = new double[columnCnt];
                double min_dis = Double.MAX_VALUE;
                for (double[] candidate : candidates) {
                    if (delta(x_repaired_predicted, candidate) < min_dis) {
                        min_dis = delta(x_repaired_predicted, candidate);
                        optimal_repair = candidate;
                    }
                }
                this.td_repaired[i] = optimal_repair;
            } else {
//                this.td_prediction[i] = td[i];
                this.td_repaired[i] = td[i];
            }
//            if (delta(x_repaired_predicted, optimal_repair) > beta) {
//                this.anomalies_in_repaired[i] = Boolean.TRUE;
//            } else {
//                this.anomalies_in_repaired[i] = Boolean.FALSE;
//            }
            i++;
        }
    }


    public void backwardRepairing(int p) {
        int i = initial_window[0] - 1;
        if (i < 0) {
            return;
        }

        while (i >= 0) {
            if (td_anomalies[i] == Boolean.TRUE) {
                double[] optimal_repair = new double[columnCnt];
//                double[] optimal_prediction_repair = new double[columnCnt];
                double[][] candidates =
                        this.kdTreeUtil.kNearestNeighbors(this.td_repaired[i + 1], k);
                double[][] W_repaired = getWindow(this.td_repaired, i + p, p);
                double[] x_repaired = this.td_repaired[i];
//        find the optimal repair
                double min_dis = Double.MAX_VALUE;
                for (double[] candidate : candidates) {
                    W_repaired[0] = candidate;
                    double[] x_repaired_predicted = arrayToList(prediction_model.predict(W_repaired));

                    if (delta(x_repaired_predicted, x_repaired) < min_dis) {
                        min_dis = delta(x_repaired_predicted, x_repaired);
//                        optimal_prediction_repair = x_repaired_predicted;
                        optimal_repair = candidate;
                    }
                }
                this.td_repaired[i] = optimal_repair;
//                this.td_prediction[i] = optimal_prediction_repair;
            } else {
//                optimal_repair = td[i];
//                this.td_prediction[i] = optimal_repair;
                this.td_repaired[i] = td[i];
            }

//            if (delta(optimal_prediction_repair, td[i + p]) > beta) {
//                this.anomalies_in_repaired[i] = Boolean.TRUE;
//            } else {
//                this.anomalies_in_repaired[i] = Boolean.FALSE;
//            }
            i--;
        }
    }

    public void initList() {
//        td_prediction = new double[n][columnCnt];
        td_repaired = new double[n][columnCnt];
//        anomalies_in_repaired = new boolean[n];
    }

    public void repair() {
        call_std();
        getOriginalAnomaliesAndLearnModel();
        findInitialWindow(p);
        initList();
        System.arraycopy(td, 0, td_repaired, 0, initial_window[1] + 1);
        backwardRepairing(p);
        forwardRepairing(p);
    }

    public double[] arrayToList(ArrayList<Double> arrayList) {
        double[] list = new double[arrayList.size()];
        for (int i = 0; i < arrayList.size(); i++) {
            list[i] = arrayList.get(i);
        }
        return list;
    }

    public double[][] getTd_repaired() {
        return td_repaired;
    }

//    public boolean[] getAnomalies_in_repaired() {
//        return anomalies_in_repaired;
//    }

    public long getCost_time() {
        return cost_time;
    }
}