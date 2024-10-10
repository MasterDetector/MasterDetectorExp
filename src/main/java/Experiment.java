import Algorithm.*;
import Algorithm.util.KDTreeUtil;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

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
            td_len = 600 * 1000;
            md_len = 20000;

            error_rate = 5;
            error_range = 3.0;
            error_length = 5;
        } else if (dataset_idx == 1) {
            td_path = "./data/gps/time_series_data_1166375.csv";
            md_path = "./data/gps/master_data_92110.csv";
            td_len = 200 * 1000;
            md_len = 80000;

            error_rate = 1;
            error_range = 0.8;
            error_length = 5;
        } else if (dataset_idx == 2) {
            td_path = "./data/road/time_series_data_1829660.csv";
            md_path = "./data/road/master_data_66876.csv";
            td_len = 300 * 1000;
            md_len = 6000;

            error_rate = 5;
            error_range = 3.0;
            error_length = 5;
        } else if (dataset_idx == 3) {
            td_path = "./data/weather/time_series_data_390598.csv";
            md_path = "./data/weather/master_data_524288.csv";
            td_len = 30 * 1000;
            md_len = 6000;

            error_rate = 5;
            error_range = 3.0;
            error_length = 5;
        } else {
            td_path = "./data/traj/time_series_data_25168.csv";
            md_path = "./data/traj/master_data_1180.csv";
            td_len = 25 * 1000;
            md_len = 1200;

            error_rate = 5;
            error_range = 3.0;
            error_length = 5;
        }
        // 4imr
        label_rate = 0.2;
        // 4master
        k = 3;
        p = 10;
        eta = 0.1;
        beta = 0.1;
        // random
        seed = 665;
    }

    public static Analysis masterRepair(KDTreeUtil kdTree, long[] td_time, double[][] td_clean, double[][] td_dirty, boolean[] td_bool, boolean[] detect_clean) {
        System.out.println("\nMasterRepair");
        int columnCnt = td_clean[0].length;
        MasterDetector masterRepair = new MasterDetector(td_dirty, kdTree, td_time, columnCnt, k, p, eta);
        double[][] td_repair = masterRepair.getTd_repaired();
        double regression_loss = masterRepair.getRegression_loss();
//        saveTrajectory(td_repair);
        long cost_time = masterRepair.getCost_time();
        boolean[] detect_repair = AnomalyDetector.detect(td_repair, p, beta);
        return new Analysis(td_time, td_clean, td_repair, td_bool, cost_time, detect_clean, detect_repair);
    }

    public static void writeDataToCSV(String[] td_time, double[][] data, String csvFile) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvFile))) {
//            bw.write(header);
//            bw.newLine();
            for (int i = 0; i < data.length; i++) {
                double[] row = data[i];
                String time = td_time[i];
                StringBuilder line = new StringBuilder();
//                System.out.println(time);
                line.append(time);
                for (double value : row) {
                    if (line.length() > 0) {
                        line.append(",");
                    }
                    line.append(value);
                }
                bw.write(line.toString());
                bw.newLine();
            }
        } catch (IOException e) {
            System.err.println("ERROR: " + e.getMessage());
        }
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

    public static void main(String[] args) throws Exception {
        starts();
        main_td_scale();
//        main_md_scale();
//        main_error_rate();
//        main_error_range();
//        main_error_length();
//        main_parameters(false, true, false, false);
//        whole_data_set(4);
//        varyingModel(5);
    }


    public static void starts() throws Exception {
        SimpleDateFormat df = new SimpleDateFormat("yyyy_MM_dd_HH_MM_SS");
        String current_time = df.format(new Date());
        System.out.println("current time " + current_time);
        recordFile("\n" + current_time + "\n", "RMSE");
        recordFile("\n" + current_time + "\n", "Precision");
        recordFile("\n" + current_time + "\n", "Recall");
        recordFile("\n" + current_time + "\n", "Time");
        recordFile("\n" + current_time + "\n", "RegressionLoss");
    }

    public static void varyingModel(int error_rate) throws Exception {
        recordFile("\nERROR RATE " + error_rate + "\n", "RMSE");
        recordFile("\nERROR RATE " + error_rate + "\n", "Time");
        recordFile("\nERROR RATE " + error_rate + "\n", "Precision");
        recordFile("\nERROR RATE " + error_rate + "\n", "Recall");
        recordFile("\nERROR RATE " + error_rate + "\n", "RegressionLoss");
        String data_csv_path = "./model/data/";
        String[] datasets = {"engine", "gps", "road", "weather", "traj"};
        for (int dataset_idx = 0; dataset_idx < 5; dataset_idx++) {
            String file_path = data_csv_path + datasets[dataset_idx] + "/";
            File directory = new File(file_path);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            init(dataset_idx);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            String [] td_time_str = loadData.getTd_time_str();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
//            System.out.println(Arrays.toString(td_time));
            writeDataToCSV(td_time_str, td_clean, file_path + "clean_t.csv");

            // detect ground truth
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            // add noise
            AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
            double[][] td_dirty = addNoise.getTd_dirty();
            writeDataToCSV(td_time_str, td_dirty, file_path + "dirty_" + error_rate + "_t.csv");

            // label4imr
            LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
            boolean[] td_bool = labelData.getTd_bool();

            recordFile("\n" + datasets[dataset_idx] + ",", "RMSE");
            recordFile("\n" + datasets[dataset_idx] + ",", "Precision");
            recordFile("\n" + datasets[dataset_idx] + ",", "Recall");
            recordFile("\n" + datasets[dataset_idx] + ",", "Time");
            recordFile("\n" + datasets[dataset_idx] + ",", "RegressionLoss");

            System.out.println("\nMasterRepair");
            int columnCnt = td_clean[0].length;
            MasterDetector masterRepair = new MasterDetector(td_dirty, kdTree, td_time, columnCnt, k, p, eta);
            double[][] td_repair_mr = masterRepair.getTd_repaired();
            writeDataToCSV(td_time_str, td_repair_mr, file_path + "VAR_t.csv");

            long cost_time = masterRepair.getCost_time();
            boolean[] detect_repair = AnomalyDetector.detect(td_repair_mr, p, beta);
            Analysis analysis = new Analysis(td_time, td_clean, td_repair_mr, td_bool, cost_time, detect_clean, detect_repair);
            double regression_loss = masterRepair.getRegression_loss();
            recordFile(analysis.getRMSE() + ",", "RMSE");
            recordFile(analysis.getPrecision() + ",", "Precision");
            recordFile(analysis.getRecall() + ",", "Recall");
            recordFile(analysis.getCost_time() + ",", "Time");
            recordFile(regression_loss + ",", "RegressionLoss");

            System.gc();
            Runtime.getRuntime().gc();
        }
    }

    public static double[][] loadCsvFile(String filePath) throws FileNotFoundException {
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        Scanner sc = new Scanner(new File(filePath));
        sc.useDelimiter("\\s*([,\\r\\n])\\s*"); // set separator
        sc.nextLine();  // skip table header
        for (int k = td_len; k > 0 && sc.hasNextLine(); --k) {  // the size of td_clean is dataLen
            String new_line = sc.nextLine();

            if (new_line.charAt(new_line.length()-1) == ',') {
                new_line = new_line + '0';
            }
            String[] line_str = (new_line).split(",");

            addValues(data, line_str);
        }


        double[][] data_array = getDoubleArray(data);
        return data_array;
    }

    private static void addValues(ArrayList<ArrayList<Double>> array, String[] line) {
        ArrayList<Double> values = new ArrayList<>();
        String value;
        for (int i = 1; i < line.length; ++i) {
            value = line[i];
            if (!value.equals("")) {
                values.add(Double.parseDouble(value));
            } else {
                values.add(Double.NaN);
            }
        }
        array.add(values);
    }

    private static double[][] getDoubleArray(ArrayList<ArrayList<Double>> arrayList) {
        double[][] rtn = new double[arrayList.size()][arrayList.get(0).size()];
        for (int i = 0, j; i < arrayList.size(); ++i)
            for (j = 0; j < arrayList.get(0).size(); ++j)
                rtn[i][j] = arrayList.get(i).get(j);
        return rtn;
    }

    public static double[][] trimToShortestLength(double[][] list1, double[][] list2) {
        int minLength = Math.min(list1.length, list2.length);
        double[][] trimmedList = new double[minLength][];
        System.arraycopy(list1, 0, trimmedList, 0, minLength);
        return trimmedList;
    }

    public static void checkLSTMPrecisionAndRecall(int error_rate) throws Exception {
        recordFile("\nLSTM ERROR RATE " + error_rate + "\n", "Precision");
        recordFile("\nLSTM ERROR RATE " + error_rate + "\n", "Recall");

        String data_csv_path = "./LSTM/repaired_data/";
        String clean_file_path = "./model/data/";
        String[] datasets = {"engine", "gps", "road", "weather", "traj"};
        for (int dataset_idx = 0; dataset_idx < 5; dataset_idx++) {
            String file_path = data_csv_path + datasets[dataset_idx] + "_" + String.valueOf(error_rate) + ".csv";
            String clean_path = clean_file_path + datasets[dataset_idx] + "/clean.csv";

            init(dataset_idx);

            double[][] clean_data = loadCsvFile(clean_path);
            double[][] repaired_data = loadCsvFile(file_path);

            clean_data = trimToShortestLength(clean_data, repaired_data);
            repaired_data = trimToShortestLength(repaired_data, clean_data);


            boolean[] detect_clean = AnomalyDetector.detect(clean_data, p, beta);
            boolean[] detect_repair = AnomalyDetector.detect(repaired_data, p, beta);

            recordFile("\n" + datasets[dataset_idx] + ",", "Precision");
            recordFile("\n" + datasets[dataset_idx] + ",", "Recall");


            int tp = 0, fp = 0, fn = 0;

            for (int i = 0; i < detect_repair.length; i++) {
                if (detect_repair[i] && detect_clean[i]) {
                    tp++;
                } else if (detect_repair[i] && !detect_clean[i]) {
                    fp++;
                } else if (!detect_repair[i] && detect_clean[i]) {
                    fn++;
                }
            }
            tp++;
            fp++;
            fn++;

            double precision = tp / (double) (tp + fp);
            double recall = tp / (double) (tp + fn);
            double F1score = (2 * precision * recall) / (precision + recall);

            recordFile(precision + ",", "Precision");
            recordFile(recall + ",", "Recall");

            System.gc();
            Runtime.getRuntime().gc();
        }
    }

    public static void get_data_repaired(int rate) throws Exception {
        error_rate = rate;
        String data_csv_path = "./data/error_rate_" + String.valueOf(rate) + "/";
        String[] datasets = {"engine", "gps", "road", "weather", "traj"};
        for (int dataset_idx = 4; dataset_idx < 5; dataset_idx++) {
            String file_path = data_csv_path + datasets[dataset_idx] + "/";
            // 创建File对象
            File directory = new File(file_path);

            // 判断文件夹是否存在
            if (!directory.exists()) {
                // 文件夹不存在，创建文件夹
                directory.mkdirs();
            }
            init(dataset_idx);

            LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
            long[] td_time = loadData.getTd_time();
            double[][] td_clean = loadData.getTd_clean();
            double[][] md = loadData.getMd();
            KDTreeUtil kdTree = loadData.getKdTree();
            KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
//            writeDataToCSV(td_clean, file_path + "clean.csv");

            // detect ground truth
            boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

            // add noise
            AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
            double[][] td_dirty = addNoise.getTd_dirty();
//            writeDataToCSV(td_dirty, file_path + "dirty.csv");

            // label4imr
            LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
            double[][] td_label = labelData.getTd_label();
            boolean[] td_bool = labelData.getTd_bool();

            recordFile("\n\n", "RMSE");
            recordFile("\n\n", "Precision");
            recordFile("\n\n", "Recall");
            recordFile("\n\n", "Time");

            System.out.println("\nMasterRepair");
            int columnCnt = td_clean[0].length;
            MasterDetector masterRepair = new MasterDetector(td_dirty, kdTree, td_time, columnCnt, k, p, eta);
            double[][] td_repair_mr = masterRepair.getTd_repaired();
//            writeDataToCSV(td_repair_mr, file_path + "MR.csv");
            long cost_time = masterRepair.getCost_time();
            boolean[] detect_repair = AnomalyDetector.detect(td_repair_mr, p, beta);
            Analysis analysis = new Analysis(td_time, td_clean, td_repair_mr, td_bool, cost_time, detect_clean, detect_repair);
            recordFile(analysis.getRMSE() + ",", "RMSE");
            recordFile(analysis.getPrecision() + ",", "Precision");
            recordFile(analysis.getRecall() + ",", "Recall");
            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nERRepair");
            EditingRuleRepair editingRuleRepair = new EditingRuleRepair(td_time, td_dirty, md);
            double[][] td_repair_er = editingRuleRepair.getTd_repair();
//            writeDataToCSV(td_repair_er, file_path + "ER.csv");
            cost_time = editingRuleRepair.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_er, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_er, td_bool, cost_time, detect_clean, detect_repair);
            recordFile(analysis.getRMSE() + ",", "RMSE");
            recordFile(analysis.getPrecision() + ",", "Precision");
            recordFile(analysis.getRecall() + ",", "Recall");
            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nSCREEN");
            SCREEN screen = new SCREEN(td_time, td_dirty);
            double[][] td_repair_sr = screen.getTd_repair();
//            writeDataToCSV(td_repair_sr, file_path + "SCREEN.csv");
            cost_time = screen.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_sr, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_sr, td_bool, cost_time, detect_clean, detect_repair);
            recordFile(analysis.getRMSE() + ",", "RMSE");
            recordFile(analysis.getPrecision() + ",", "Precision");
            recordFile(analysis.getRecall() + ",", "Recall");
            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nLsgreedy");
            Lsgreedy lsgreedy = new Lsgreedy(td_time, td_dirty);
            double[][] td_repair_lg = lsgreedy.getTd_repair();
//            writeDataToCSV(td_repair_lg, file_path + "LG.csv");
            cost_time = lsgreedy.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_lg, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_lg, td_bool, cost_time, detect_clean, detect_repair);
            recordFile(analysis.getRMSE() + ",", "RMSE");
            recordFile(analysis.getPrecision() + ",", "Precision");
            recordFile(analysis.getRecall() + ",", "Recall");
            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

            System.out.println("\nIMR");
            IMR imr = new IMR(td_time, td_dirty, td_label, td_bool);
            double[][] td_repair_imr = imr.getTd_repair();
//            writeDataToCSV(td_repair_imr, file_path + "IMR.csv");
            cost_time = imr.getCost_time();
            detect_repair = AnomalyDetector.detect(td_repair_imr, p, beta);
            analysis = new Analysis(td_time, td_clean, td_repair_imr, td_bool, cost_time, detect_clean, detect_repair);
            recordFile(analysis.getRMSE() + ",", "RMSE");
            recordFile(analysis.getPrecision() + ",", "Precision");
            recordFile(analysis.getRecall() + ",", "Recall");
            recordFile(analysis.getCost_time() + ",", "Time");
            System.gc();
            Runtime.getRuntime().gc();

        }
    }

    public static void whole_data_set(int dataset_idx) throws Exception {
        init(dataset_idx);
        String record = "traj_td-scale_whole\n";
        recordFile(record, "RMSE");
        recordFile(record, "Precision");
        recordFile(record, "Recall");
        recordFile(record, "Time");

        LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
        long[] td_time = loadData.getTd_time();
        double[][] td_clean = loadData.getTd_clean();
        double[][] td_dirty = loadData.getTd_raw_array();
        double[][] md = loadData.getMd();
        KDTreeUtil kdTree = loadData.getKdTree();
        KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
        System.out.println("finish loading data");

        // detect ground truth
        boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

        // label4imr
        LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
        double[][] td_label = labelData.getTd_label();
        boolean[] td_bool = labelData.getTd_bool();

        boolean[] default_bool = new boolean[td_bool.length];
        Arrays.fill(default_bool, false);
        System.out.println("finish labeling data");

        Analysis analysis;
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
        recordFile("\n\n", "RMSE");
        recordFile("\n\n", "Precision");
        recordFile("\n\n", "Recall");
        recordFile("\n\n", "Time");
    }

    public static void main_td_scale() throws Exception {
        System.out.println("--------------------");
        System.out.println("main td scale");
        System.out.println("--------------------");
        int td_len_base, td_len_step;
        String record;
        Analysis analysis;
        for (int dataset_idx = 4; dataset_idx < 5; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                td_len_base = 600 * 1000;
                td_len_step = 100 * 1000;
                record = "fuel_td-scale_0.6m,0.7m,0.8m,0.9m,1.0m,1.1m,1.2m,1.3m,1.4m,1.5m\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                td_len_base = 100 * 1000;
                td_len_step = 100 * 1000;
                record = "gps_td-scale_0.1m,0.2m,0.3m,0.4m,0.5m,0.6m,0.7m,0.8m,0.9m,1.0m\n";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                td_len_base = 600 * 1000;
                td_len_step = 100 * 1000;
                record = "road_td-scale_0.6m,0.7m,0.8m,0.9m,1.0m,1.1m,1.2m,1.3m,1.4m,1.5m\n";
            } else if (dataset_idx == 3)  {
                init(dataset_idx);
                td_len_base = 60 * 1000;
                td_len_step = 10 * 1000;
                record = "weather_td-scale_60k,70k,80k,90k,100k,120k,140k,160k,180k,200k\n";
            } else {
                init(dataset_idx);
                td_len_base = 2500;
                td_len_step = 2500;
                record = "traj_td-scale_2.5k,5k,7.5k,10k,12.5k,15k,17.5k,20k,22.5k,25k\n";
            }
            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            for (td_len = td_len_base; td_len <= td_len_base + td_len_step * 0; td_len += td_len_step) {
                System.out.println("td_len "+ td_len);
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
                long[] td_time = loadData.getTd_time();
                double[][] td_clean = loadData.getTd_clean();
                double[][] md = loadData.getMd();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
                System.out.println("finish loading data");

                // detect ground truth
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
                double[][] td_dirty = addNoise.getTd_dirty();
                System.out.println("finish adding noise");

                // label4imr
                LabelData labelData = new LabelData(td_clean, td_dirty, label_rate, seed);
                double[][] td_label = labelData.getTd_label();
                boolean[] td_bool = labelData.getTd_bool();

                boolean[] default_bool = new boolean[td_bool.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish labeling data");

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
                recordFile("\n\n", "RMSE");
                recordFile("\n\n", "Precision");
                recordFile("\n\n", "Recall");
                recordFile("\n\n", "Time");
            }
        }
    }

    public static void main_md_scale() throws Exception {
        System.out.println("--------------------");
        System.out.println("main md scale");
        System.out.println("--------------------");
        int md_len_base, md_len_step;
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                md_len_base = 5000;
                md_len_step = 5000;
                record = "fuel_md-scale_5k,10k,15k,20k,25k,30k,35k,40k,45k,50k\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                md_len_base = 5000;
                md_len_step = 5000;
                record = "gps_md-scale_2k,4k,6k,8k,10k,12k,14k,16k,18k,20k\n";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                md_len_base = 2000;
                md_len_step = 2000;
                record = "road_md-scale_1k,10k,20k,30k,40k,50k,60k,70k,80k,90k\n";
            } else {
                init(dataset_idx);
                md_len_base = 2000;
                md_len_step = 2000;
                record = "weather_md-scale_2k,4k,6k,8k,10k,12k,14k,16k,18k,20k\n";
            }
            recordFile(record, "RMSE");
            recordFile(record, "Precision");
            recordFile(record, "Recall");
            recordFile(record, "Time");

            for (md_len = md_len_base; md_len <= md_len_base + md_len_step * 9; md_len += md_len_step) {
                System.out.println("md_len "+ md_len);
                LoadData loadData = new LoadData(td_path, md_path, td_len, md_len, eta, seed);
                long[] td_time = loadData.getTd_time();
                double[][] td_clean = loadData.getTd_clean();
                double[][] md = loadData.getMd();
                KDTreeUtil kdTree = loadData.getKdTree();
                KDTreeUtil kdTreeComplete = loadData.getKdTreeComplete();
                System.out.println("finish loading data");

                // detect ground truth
                boolean[] detect_clean = AnomalyDetector.detect(td_clean, p, beta);

                // add noise
                AddNoise addNoise = new AddNoise(td_clean, error_rate, error_range, error_length, eta, kdTreeComplete, seed);
                double[][] td_dirty = addNoise.getTd_dirty();
                System.out.println("finish adding noise");

                // label
                boolean[] default_bool = new boolean[td_clean.length];
                Arrays.fill(default_bool, false);
                System.out.println("finish labeling data");

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
        System.out.println("--------------------");
        System.out.println("main error rate");
        System.out.println("--------------------");
        int error_rate_base, error_rate_step;
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                error_rate_base = 6;
                error_rate_step = 1;
                record = "fuel_error-rate_0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                error_rate_base = 1;
                error_rate_step = 1;
                record = "gps_error-rate_0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                error_rate_base = 6;
                error_rate_step = 1;
                record = "road_error-rate_0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5\n";
            } else {
                init(dataset_idx);
                error_rate_base = 6;
                error_rate_step = 1;
                record = "weather_error-rate_0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5\n";
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

            for (error_rate = error_rate_base; error_rate <= error_rate_base + 9 * error_rate_step; error_rate += error_rate_step) {
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
        System.out.println("--------------------");
        System.out.println("main error range");
        System.out.println("--------------------");
        double error_range_base, error_range_step;
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                error_range_base = 0.5;
                error_range_step = 0.5;
                record = "fuel_error-range_0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                error_range_base = 0.2;
                error_range_step = 0.2;
                record = "gps_error-range_0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0\n";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                error_range_base = 0.5;
                error_range_step = 0.5;
                record = "road_error-range_0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0\n";
            } else {
                init(dataset_idx);
                error_range_base = 0.5;
                error_range_step = 0.5;
                record = "weather_error-range_0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0\n";
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

            for (error_range = error_range_base; error_range <= error_range_base + error_range_step * 9; error_range += error_range_step) {
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
        System.out.println("--------------------");
        System.out.println("main error length");
        System.out.println("--------------------");
        String record;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                record = "fuel_error-length_1,2,3,4,5,6,7,8,9,10\n";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                record = "gps_error-length_1,2,3,4,5,6,7,8,9,10\n";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                record = "road_error-length_1,2,3,4,5,6,7,8,9,10\n";
            } else {
                init(dataset_idx);
                record = "weather_error-length_1,2,3,4,5,6,7,8,9,10\n";
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

            for (error_length = 1; error_length <= 10; error_length += 1) {
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
        System.out.println("--------------------");
        System.out.println("main parameters");
        System.out.println("--------------------");
        String record, parameter_type;
        Analysis analysis;
        for (int dataset_idx = 0; dataset_idx < 4; dataset_idx++) {
            System.out.println("--------------------");
            System.out.println("in dataset " + dataset_idx);
            System.out.println("--------------------");
            if (dataset_idx == 0) {
                init(dataset_idx);
                record = "fuel_";
            } else if (dataset_idx == 1) {
                init(dataset_idx);
                record = "gps_";
            } else if (dataset_idx == 2) {
                init(dataset_idx);
                record = "road_";
            } else {
                init(dataset_idx);
                record = "weather_";
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
                parameter_type = "eta_0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0\n";
                recordFile(record + parameter_type, "RMSE");
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                recordFile(record + parameter_type, "Time");
                for (eta = 0.2; eta <= 2.0; eta += 0.2) {
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
                parameter_type = "p_5,10,15,20,25\n";
                recordFile(record + parameter_type, "RMSE");
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                recordFile(record + parameter_type, "Time");
                for (p = 5; p <= 25; p += 5) {
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
                parameter_type = "beta_0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5\n";
                recordFile(record + parameter_type, "Precision");
                recordFile(record + parameter_type, "Recall");
                recordFile(record + parameter_type, "RMSE");
                recordFile(record + parameter_type, "Time");
                for (beta = 0.25; beta <= 2.5; beta += 0.25) {
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
                        recordFile(analysis.getRMSE() + ",", "RMSE");
                        recordFile(analysis.getCost_time() + ",", "Time");
                        System.gc();
                        Runtime.getRuntime().gc();
                    }
                    recordFile("\n", "Precision");
                    recordFile("\n", "Recall");
                    recordFile("\n", "RMSE");
                    recordFile("\n", "Time");
                }
            }
        }
    }
}
