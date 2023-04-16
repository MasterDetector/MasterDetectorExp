import Algorithm.*;
import Algorithm.util.KDTreeUtil;

import java.util.Arrays;
import java.io.FileWriter;
import java.io.BufferedWriter;

public class Experiment {
    // input
    static String td_path;
    static String md_path;
    static int td_len;
    static int md_len;
    // error
    static int error_rate;
    static double error_range;
    static int error_length;
    // 4imr
    static double label_rate;
    // 4master
    static int k;
    static int p;
    static double eta;
    static double beta;
    // random
    static int seed;

    public static void init(int dataset_idx) {
        if (dataset_idx == 0) {
            td_path = "./data/engine/time_series_data_1596148.csv";
            md_path = "./data/engine/master_data_14756.csv";
            td_len = 300 * 1000;
            md_len = 3000;

            error_rate = 5;
            error_range = 3.0;
            error_length = 5;
        } else if (dataset_idx == 1) {
            td_path = "./data/gps/time_series_data_1166375.csv";
            md_path = "./data/gps/master_data_92110.csv";
            td_len = 200 * 1000;
            md_len = 8000;

            error_rate = 1;
            error_range = 0.8;
            error_length = 5;
        } else {
            td_path = "./data/road/time_series_data_1829660.csv";
            md_path = "./data/road/master_data_66876.csv";
            td_len = 300 * 1000;
            md_len = 6000;

            error_rate = 5;
            error_range = 3.0;
            error_length = 5;
        }
        // 4imr
        label_rate = 0.2;
        // 4master
        k = 3;
        p = 3;
        eta = 1.0;
        beta = 1.0;
        // random
        seed = 665;
    }

    public static Analysis masterRepair(KDTreeUtil kdTree, long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nMasterRepair");
        int columnCnt = td_clean[0].length;
        MasterDetector masterRepair = new MasterDetector(td_dirty, kdTree, td_time, columnCnt, k, p, eta);
        double[][] td_repair = masterRepair.getTd_repaired();
        long cost_time = masterRepair.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis erRepair(double[][] md, long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nERRepair");
        EditingRuleRepair editingRuleRepair = new EditingRuleRepair(td_time, td_dirty, md);
        double[][] td_repair = editingRuleRepair.getTd_repair();
        long cost_time = editingRuleRepair.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis screenRepair(long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) throws Exception {
        System.out.println("\nSCREEN");
        SCREEN screen = new SCREEN(td_time, td_dirty);
        double[][] td_repair = screen.getTd_repair();
        long cost_time = screen.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis lsgreedyRepair(long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) throws Exception {
        System.out.println("\nLsgreedy");
        Lsgreedy lsgreedy = new Lsgreedy(td_time, td_dirty);
        double[][] td_repair = lsgreedy.getTd_repair();
        long cost_time = lsgreedy.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static Analysis imrRepair(long[] td_time, double[][] td_clean, double[][] td_dirty, double[][] td_label, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nIMR");
        IMR imr = new IMR(td_time, td_dirty, td_label, td_bool);
        double[][] td_repair = imr.getTd_repair();
        long cost_time = imr.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static void recordFile(String string, String type) throws Exception {
        FileWriter fileWritter = new FileWriter("./results/exp" + type + ".txt", true);
        BufferedWriter bw = new BufferedWriter(fileWritter);
        bw.write(string);
        bw.close();
    }

    public static void main(String[] args) throws Exception { //synthetic
        main_error_rate();
        main_error_range();
        main_error_length();
        main_td_scale();
        main_md_scale();
        main_parameters(false, true, false, false);
    }

    public static void main_td_scale() throws Exception {
        int td_len_base;
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 3; dataset_idx++) {
            if (dataset_idx == 0) {
                init(dataset_idx);
                td_len_base = 300 * 1000;
                record = "fuel_td-scale_0.3m,0.6m,0.9m,1.2m,1.5m\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                td_len_base = 200 * 1000;
                record = "gps_td-scale_0.2m,0.4m,0.6m,0.8m,1m\n";
            } else {
                init(dataset_idx);
                td_len_base = 300 * 1000;
                record = "road_td-scale_0.3m,0.6m,0.9m,1.2m,1.5m\n";
            }
            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            for (td_len = td_len_base; td_len <= td_len_base * 5; td_len += td_len_base) {
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
                long[] td_time = loadData.getTd_time();
                double[][] td_clean = loadData.getTd_clean();
                double[][] md = loadData.getMd();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
                System.out.println("finish load data");

                // detect ground truth
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
                double[][] td_dirty = addNoise.getTd_dirty();
                System.out.println("finish add noise");

                // label4imr
                LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                double[][] td_label = labelData.getTd_label();
                boolean[] td_bool = labelData.getTd_bool();

                boolean[] default_bool = new boolean[td_bool.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish label data");

                // repair
                for (int j = 0; j < 5; j++) {
                    if (j == 0)
                        analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 1)
                        analysis = erRepair(md, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 2)
                        analysis = screenRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 3)
                        analysis = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else
                        analysis = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool, detect_clean);
                    recordFile(analysis.getRMSE() + ",", "RMSE");
                    recordFile(analysis.getPrecision() + ",", "Precision");
                    recordFile(analysis.getRecall() + ",", "Recall");
                    recordFile(analysis.getCost_time() + ",", "Time");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                recordFile("\n", "RMSE");
                recordFile("\n", "Precision");
                recordFile("\n", "Recall");
                recordFile("\n", "Time");
            }
        }
    }

    public static void main_md_scale() throws Exception {
        int md_len_base;
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 3; dataset_idx++) {
            if (dataset_idx == 0) {
                init(dataset_idx);
                md_len_base = 100;
                record = "fuel_md-scale_0.5k,1k,1.5k,2k,2.5k\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                md_len_base = 400;
                record = "gps_md-scale_2k,4k,6k,8k,10k\n";
            } else {
                init(dataset_idx);
                md_len_base = 200;
                record = "road_md-scale_1k,2k,3k,4k,5k\n";
            }
            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            for (md_len = md_len_base; md_len <= md_len_base * 5; md_len += md_len_base) {
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
                long[] td_time = loadData.getTd_time();
                double[][] td_clean = loadData.getTd_clean();
                double[][] md = loadData.getMd();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
                System.out.println("finish load data");

                // detect ground truth
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
                double[][] td_dirty = addNoise.getTd_dirty();
                System.out.println("finish add noise");

                // label
                boolean[] default_bool = new boolean[td_clean.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish label data");

                // repair
                for (int j = 0; j < 2; j++) {
                    if (j == 0) {
                        analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    } else {
                        analysis = erRepair(md, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    }
                    recordFile(analysis.getRMSE() + ",", "RMSE");
                    recordFile(analysis.getPrecision() + ",", "Precision");
                    recordFile(analysis.getRecall() + ",", "Recall");
                    recordFile(analysis.getCost_time() + ",", "Time");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                recordFile("\n", "RMSE");
                recordFile("\n", "Precision");
                recordFile("\n", "Recall");
                recordFile("\n", "Time");
            }
        }
    }

    public static void main_error_rate() throws Exception {
        int error_rate_base;
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 3; dataset_idx++) {
            if (dataset_idx == 0) {
                init(dataset_idx);
                error_rate_base = 3;
                record = "fuel_error-rate_0.3,0.6,0.9,1.2,1.5\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                error_rate_base = 1;
                record = "gps_error-rate_0.1,0.2,0.3,0.4,0.5\n";
            } else {
                init(dataset_idx);
                error_rate_base = 3;
                record = "road_error-rate_0.3,0.6,0.9,1.2,1.5\n";
            }
            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            System.out.println("finish load data");

            // detect ground truth
            long startTime = System.currentTimeMillis();
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);
            long endTime = System.currentTimeMillis();
            System.out.println("LSTMDetector: " + (endTime - startTime));

            for (error_rate = error_rate_base; error_rate <= 5 * error_rate_base; error_rate += error_rate_base) {
                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
                double[][] td_dirty = addNoise.getTd_dirty();
                System.out.println("finish add noise");

                // label4imr
                LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                double[][] td_label = labelData.getTd_label();
                boolean[] td_bool = labelData.getTd_bool();

                boolean[] default_bool = new boolean[td_bool.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish label data");

                // repair
                for (int j = 0; j < 5; j++) {
                    if (j == 0)
                        analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 1)
                        analysis = erRepair(md, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 2)
                        analysis = screenRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 3)
                        analysis = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else
                        analysis = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool, detect_clean);
                    recordFile(analysis.getRMSE() + ",", "RMSE");
                    recordFile(analysis.getPrecision() + ",", "Precision");
                    recordFile(analysis.getRecall() + ",", "Recall");
                    recordFile(analysis.getCost_time() + ",", "Time");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                recordFile("\n", "RMSE");
                recordFile("\n", "Precision");
                recordFile("\n", "Recall");
                recordFile("\n", "Time");
            }
        }
    }

    public static void main_error_range() throws Exception {
        double error_range_base;
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 3; dataset_idx++) {
            if (dataset_idx == 0) {
                init(dataset_idx);
                error_range_base = 1;
                record = "fuel_error-range_1,2,3,4,5\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                error_range_base = 0.4;
                record = "gps_error-range_0.4,0.8,1.2,1.6,2.0\n";
            } else {
                init(dataset_idx);
                error_range_base = 1;
                record = "road_error-range_1,2,3,4,5\n";
            }

            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            System.out.println("finish load data");

            // detect ground truth
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            for (error_range = error_range_base; error_range <= error_range_base * 5; error_range += error_range_base) {
                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
                double[][] td_dirty = addNoise.getTd_dirty();
                System.out.println("finish add noise");

                // label4imr
                LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                double[][] td_label = labelData.getTd_label();
                boolean[] td_bool = labelData.getTd_bool();

                boolean[] default_bool = new boolean[td_bool.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish label data");

                // repair
                for (int j = 0; j < 5; j++) {
                    if (j == 0)
                        analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 1)
                        analysis = erRepair(md, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 2)
                        analysis = screenRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 3)
                        analysis = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else
                        analysis = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool, detect_clean);
                    recordFile(analysis.getRMSE() + ",", "RMSE");
                    recordFile(analysis.getPrecision() + ",", "Precision");
                    recordFile(analysis.getRecall() + ",", "Recall");
                    recordFile(analysis.getCost_time() + ",", "Time");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                recordFile("\n", "RMSE");
                recordFile("\n", "Precision");
                recordFile("\n", "Recall");
                recordFile("\n", "Time");
            }
        }
    }

    public static void main_error_length() throws Exception {
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 3; dataset_idx++) {
            if (dataset_idx == 0) {
                init(dataset_idx);
                record = "fuel_error-length_1,3,5,7,9\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                record = "gps_error-length_1,3,5,7,9\n";
            } else {
                init(dataset_idx);
                record = "road_error-length_1,3,5,7,9\n";
            }

            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            System.out.println("finish load data");


            // detect ground truth
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            for (error_length = 1; error_length <= 9; error_length += 2) {
                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
                double[][] td_dirty = addNoise.getTd_dirty();
                System.out.println("finish add noise");

                // label4imr
                LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                double[][] td_label = labelData.getTd_label();
                boolean[] td_bool = labelData.getTd_bool();

                boolean[] default_bool = new boolean[td_bool.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish label data");

                // repair
                for (int j = 0; j < 5; j++) {
                    if (j == 0)
                        analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 1)
                        analysis = erRepair(md, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 2)
                        analysis = screenRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else if (j == 3)
                        analysis = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                    else
                        analysis = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool, detect_clean);
                    recordFile(analysis.getRMSE() + ",", "RMSE");
                    recordFile(analysis.getPrecision() + ",", "Precision");
                    recordFile(analysis.getRecall() + ",", "Recall");
                    recordFile(analysis.getCost_time() + ",", "Time");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                recordFile("\n", "RMSE");
                recordFile("\n", "Precision");
                recordFile("\n", "Recall");
                recordFile("\n", "Time");
            }
        }
    }

    public static void main_parameters(boolean k_bool, boolean p_bool, boolean eta_bool, boolean beta_bool) throws Exception {
        String record, parameter_type;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 3; dataset_idx++) {
            if (dataset_idx == 0) {
                init(dataset_idx);
                record = "fuel_";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                record = "gps_";
            } else {
                init(dataset_idx);
                record = "road_";
            }

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
            System.out.println("finish load data");


            // detect ground truth
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            // add noise
            AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
            double[][] td_dirty = addNoise.getTd_dirty();
            System.out.println("finish add noise");

            // label4imr
            LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
            double[][] td_label = labelData.getTd_label();
            boolean[] td_bool = labelData.getTd_bool();

            boolean[] default_bool = new boolean[td_bool.length];
            Arrays.fill(default_bool, false);
            System.out.println("finish label data");

            if (k_bool) {
                init(dataset_idx);
                parameter_type = "k_1,2,3,4,5\n";
                recordFile(record + parameter_type, "RMSE");
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                recordFile(record + parameter_type, "Time");
                for (k = 1; k <= 5; k += 1) {
                    // repair
                    analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    recordFile(analysis.getRMSE() + ",\n", "RMSE");
                    recordFile(analysis.getPrecision() + ",\n", "Precision");
                    recordFile(analysis.getRecall() + ",\n", "Recall");
                    recordFile(analysis.getCost_time() + ",\n", "Time");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
            }

            if (eta_bool) {
                init(dataset_idx);
                parameter_type = "eta_0.4,0.8,1.2,1.6,2.0\n";
                recordFile(record + parameter_type, "RMSE");
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                recordFile(record + parameter_type, "Time");
                for (eta = 0.4; eta <= 2.0; eta += 0.4) {
                    // repair
                    analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                    recordFile(analysis.getRMSE() + ",\n", "RMSE");
                    recordFile(analysis.getPrecision() + ",\n", "Precision");
                    recordFile(analysis.getRecall() + ",\n", "Recall");
                    recordFile(analysis.getCost_time() + ",\n", "Time");
                    System.gc();
                    Runtime.getRuntime().gc();
                }
            }

            if (p_bool) {
                init(dataset_idx);
                parameter_type = "p_3,6,9,12,15\n";
                recordFile(record + parameter_type, "RMSE");
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                recordFile(record + parameter_type, "Time");
                for (p = 3; p <= 15; p += 3) {
                    for (int j = 0; j < 6; j++) {
                        if (j == 0)
                            analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else if (j == 1)
                            analysis = erRepair(md, td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else if (j == 2)
                            analysis = screenRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else if (j == 3)
                            analysis = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else
                            analysis = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool, detect_clean);
                        recordFile(analysis.getRMSE() + ",", "RMSE");
                        recordFile(analysis.getPrecision() + ",", "Precision");
                        recordFile(analysis.getRecall() + ",", "Recall");
                        recordFile(analysis.getCost_time() + ",", "Time");
                        System.gc();
                        Runtime.getRuntime().gc();
                    }
                    recordFile("\n", "RMSE");
                    recordFile("\n", "Precision");
                    recordFile("\n", "Recall");
                    recordFile("\n", "Time");
                }
            }

            if (beta_bool) {
                init(dataset_idx);
                parameter_type = "beta_0.5,1.0,1.5,2.0,2.5\n";
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                for (beta = 0.5; beta <= 2.5; beta += 0.5) {
                    for (int j = 0; j < 5; j++) {
                        if (j == 0)
                            analysis = masterRepair(kdTree, td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else if (j == 1)
                            analysis = erRepair(md, td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else if (j == 2)
                            analysis = screenRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else if (j == 3)
                            analysis = lsgreedyRepair(td_time, td_clean, td_dirty, default_bool, detect_clean);
                        else
                            analysis = imrRepair(td_time, td_clean, td_dirty, td_label, td_bool, detect_clean);
                        recordFile(analysis.getPrecision() + ",", "Precision");
                        recordFile(analysis.getRecall() + ",", "Recall");
                        System.gc();
                        Runtime.getRuntime().gc();
                    }
                    recordFile("\n", "Precision");
                    recordFile("\n", "Recall");
                }
            }
        }
    }
}
