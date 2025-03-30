package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Compute the conditional probability using "stripes" approach
 */
public class CORStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CORStripes.class);

    /*
     * First-pass Mapper: emits <word, 1> for each unique word in the document
     */
    private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private static final Text WORD = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            HashSet<String> wordSet = new HashSet<>();
            // Tokenize the document
            String cleanDoc = value.toString().replaceAll("[^a-zA-Z ]", " ");
            StringTokenizer docTokenizer = new StringTokenizer(cleanDoc);

            // Collect unique words in the document
            while (docTokenizer.hasMoreTokens()) {
                String word = docTokenizer.nextToken().toLowerCase();
                wordSet.add(word);
            }

            // Emit <word, 1> for each unique word
            for (String word : wordSet) {
                WORD.set(word);
                context.write(WORD, ONE);
            }
        }
    }

    /*
     * First-pass Reducer: aggregates word counts
     */
    private static class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;

            // Sum up counts for each word
            for (IntWritable value : values) {
                sum += value.get();
            }

            // Emit <word, total count>
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    /*
     * Second-pass Mapper: emits <word, stripe> where stripe is a MapWritable
     */
    public static class CORStripesMapper2 extends Mapper<LongWritable, Text, Text, MapWritable> {
        private static final MapWritable STRIPE = new MapWritable();
        private static final Text WORD = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            List<String> words = new ArrayList<>();
            // Tokenize the document
            String cleanDoc = value.toString().replaceAll("[^a-zA-Z ]", " ");
            StringTokenizer docTokenizer = new StringTokenizer(cleanDoc);

            // Collect words in the document
            while (docTokenizer.hasMoreTokens()) {
                words.add(docTokenizer.nextToken().toLowerCase());
            }

            // Generate stripes
            for (int i = 0; i < words.size(); i++) {
                STRIPE.clear();
                for (int j = 0; j < words.size(); j++) {
                    if (i != j) {
                        Text neighbor = new Text(words.get(j));
                        if (STRIPE.containsKey(neighbor)) {
                            IntWritable count = (IntWritable) STRIPE.get(neighbor);
                            count.set(count.get() + 1);
                        } else {
                            STRIPE.put(neighbor, new IntWritable(1));
                        }
                    }
                }
                WORD.set(words.get(i));
                context.write(WORD, STRIPE);
            }
        }
    }

    /*
     * Second-pass Combiner: aggregates stripes locally
     */
    public static class CORStripesCombiner2 extends Reducer<Text, MapWritable, Text, MapWritable> {
        @Override
        protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
            MapWritable combinedStripe = new MapWritable();

            // Aggregate stripes
            for (MapWritable stripe : values) {
                for (Writable neighbor : stripe.keySet()) {
                    IntWritable count = (IntWritable) stripe.get(neighbor);
                    if (combinedStripe.containsKey(neighbor)) {
                        IntWritable combinedCount = (IntWritable) combinedStripe.get(neighbor);
                        combinedCount.set(combinedCount.get() + count.get());
                    } else {
                        combinedStripe.put(neighbor, new IntWritable(count.get()));
                    }
                }
            }

            // Emit the combined stripe
            context.write(key, combinedStripe);
        }
    }

    /*
     * Second-pass Reducer: calculates conditional probabilities
     */
    public static class CORStripesReducer2 extends Reducer<Text, MapWritable, PairOfStrings, DoubleWritable> {
        private static final Map<String, Integer> WORD_TOTAL_MAP = new HashMap<>();
        private static final DoubleWritable CONDITIONAL_PROB = new DoubleWritable();
        private static final PairOfStrings BIGRAM = new PairOfStrings();

        /*
         * Preload the middle result file containing word frequencies
         */
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path middleResultPath = new Path("mid/part-r-00000");
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);

            if (!fs.exists(middleResultPath)) {
                throw new IOException(middleResultPath.toString() + " does not exist!");
            }

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(middleResultPath)))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\t");
                    WORD_TOTAL_MAP.put(parts[0], Integer.parseInt(parts[1]));
                }
            }
        }

        /*
         * Calculate conditional probabilities
         */
        @Override
        protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
            MapWritable combinedStripe = new MapWritable();

            // Aggregate stripes
            for (MapWritable stripe : values) {
                for (Writable neighbor : stripe.keySet()) {
                    IntWritable count = (IntWritable) stripe.get(neighbor);
                    if (combinedStripe.containsKey(neighbor)) {
                        IntWritable combinedCount = (IntWritable) combinedStripe.get(neighbor);
                        combinedCount.set(combinedCount.get() + count.get());
                    } else {
                        combinedStripe.put(neighbor, new IntWritable(count.get()));
                    }
                }
            }

            // Calculate conditional probabilities
            int totalWordCount = WORD_TOTAL_MAP.getOrDefault(key.toString(), 0);
            for (Writable neighbor : combinedStripe.keySet()) {
                int count = ((IntWritable) combinedStripe.get(neighbor)).get();
                double conditionalProbability = (double) count / totalWordCount;

                BIGRAM.set(key.toString(), neighbor.toString());
                CONDITIONAL_PROB.set(conditionalProbability);
                context.write(BIGRAM, CONDITIONAL_PROB);
            }
        }
    }

    /**
     * Creates an instance of this tool.
     */
    public CORStripes() {
    }

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    /**
     * Runs this tool.
     */
    @SuppressWarnings({ "static-access" })
    public int run(String[] args) throws Exception {
        Options options = new Options();

        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg()
                .withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();

        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: "
                    + exp.getMessage());
            return -1;
        }

        // Lack of arguments
        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String middlePath = "mid";
        String outputPath = cmdline.getOptionValue(OUTPUT);

        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + CORStripes.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - middle path: " + middlePath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // Setup for the first-pass MapReduce
        Configuration conf1 = new Configuration();

        Job job1 = Job.getInstance(conf1, "Firstpass");

        job1.setJarByClass(CORStripes.class);
        job1.setMapperClass(CORMapper1.class);
        job1.setReducerClass(CORReducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        FileInputFormat.setInputPaths(job1, new Path(inputPath));
        FileOutputFormat.setOutputPath(job1, new Path(middlePath));

        // Delete the output directory if it exists already.
        Path middleDir = new Path(middlePath);
        FileSystem.get(conf1).delete(middleDir, true);

        // Time the program
        long startTime = System.currentTimeMillis();
        job1.waitForCompletion(true);
        LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime)
                / 1000.0 + " seconds");

        // Setup for the second-pass MapReduce

        // Delete the output directory if it exists already.
        Path outputDir = new Path(outputPath);
        FileSystem.get(conf1).delete(outputDir, true);


        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "Secondpass");

        job2.setJarByClass(CORStripes.class);
        job2.setMapperClass(CORStripesMapper2.class);
        job2.setCombinerClass(CORStripesCombiner2.class);
        job2.setReducerClass(CORStripesReducer2.class);

        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(MapWritable.class);
        job2.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job2, new Path(inputPath));
        FileOutputFormat.setOutputPath(job2, new Path(outputPath));

        // Time the program
        startTime = System.currentTimeMillis();
        job2.waitForCompletion(true);
        LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime)
                / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new CORStripes(), args);
    }
}
