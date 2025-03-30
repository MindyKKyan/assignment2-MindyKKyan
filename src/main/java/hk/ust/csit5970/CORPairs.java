package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.StringTokenizer;
import java.util.HashMap;

/**
 * Compute the conditional probability using "pairs" approach
 */
public class CORPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CORPairs.class);

    /*
     * First-pass Mapper: emits <word, 1> for each word in the document
     */
    private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private static final Text WORD = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            HashMap<String, Integer> wordSet = new HashMap<>();
            // Tokenize the document
            String cleanDoc = value.toString().replaceAll("[^a-zA-Z ]", " ");
            StringTokenizer docTokenizer = new StringTokenizer(cleanDoc);

            // Count unique words in the document
            while (docTokenizer.hasMoreTokens()) {
                String word = docTokenizer.nextToken().toLowerCase();
                wordSet.put(word, 1);
            }

            // Emit <word, 1> for each unique word
            for (String word : wordSet.keySet()) {
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
     * Second-pass Mapper: emits <(word1, word2), 1> for each bigram in the document
     */
    public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private static final PairOfStrings BIGRAM = new PairOfStrings();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Tokenize the document
            StringTokenizer docTokenizer = new StringTokenizer(value.toString().replaceAll("[^a-zA-Z ]", " "));

            // Generate bigrams
            String prevWord = null;
            while (docTokenizer.hasMoreTokens()) {
                String currWord = docTokenizer.nextToken().toLowerCase();
                if (prevWord != null) {
                    BIGRAM.set(prevWord, currWord);
                    context.write(BIGRAM, ONE);
                }
                prevWord = currWord;
            }
        }
    }

    /*
     * Second-pass Combiner: aggregates bigram counts locally
     */
    private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;

            // Sum up counts for each bigram
            for (IntWritable value : values) {
                sum += value.get();
            }

            // Emit <bigram, total count>
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    /*
     * Second-pass Reducer: calculates conditional probabilities
     */
    public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
        private final static Map<String, Integer> wordTotalMap = new HashMap<>();
        private static final DoubleWritable CONDITIONAL_PROB = new DoubleWritable();

        /*
         * Preload the middle result file containing word frequencies
         */
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path middleResultPath = new Path("mid/part-r-00000");
            Configuration middleConf = new Configuration();
            FileSystem fs = FileSystem.get(URI.create(middleResultPath.toString()), middleConf);

            if (!fs.exists(middleResultPath)) {
                throw new IOException(middleResultPath.toString() + " does not exist!");
            }

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(middleResultPath)))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] lineTerms = line.split("\t");
                    wordTotalMap.put(lineTerms[0], Integer.parseInt(lineTerms[1]));
                }
            }
        }

        /*
         * Calculate conditional probabilities
         */
        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int bigramCount = 0;

            // Sum up counts for the bigram
            for (IntWritable value : values) {
                bigramCount += value.get();
            }

            // Get the total count of the first word in the bigram
            String firstWord = key.getLeftElement();
            if (wordTotalMap.containsKey(firstWord)) {
                int totalWordCount = wordTotalMap.get(firstWord);
                double conditionalProbability = (double) bigramCount / totalWordCount;

                // Emit <bigram, conditional probability>
                CONDITIONAL_PROB.set(conditionalProbability);
                context.write(key, CONDITIONAL_PROB);
            }
        }
    }

    private static final class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
        @Override
        public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
        }
    }

    /**
     * Creates an instance of this tool.
     */
    public CORPairs() {
    }

    private static final String INPUT = "input";
    private static final String MIDDLE = "middle";
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

        LOG.info("Tool: " + CORPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // Setup for the first-pass MapReduce
        Configuration conf1 = new Configuration();

        Job job1 = Job.getInstance(conf1, "Firstpass");

        job1.setJarByClass(CORPairs.class);
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

        job2.setJarByClass(CORPairs.class);
        job2.setMapperClass(CORPairsMapper2.class);
        job2.setCombinerClass(CORPairsCombiner2.class);
        job2.setReducerClass(CORPairsReducer2.class);

        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setMapOutputValueClass(IntWritable.class);
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
        ToolRunner.run(new CORPairs(), args);
    }
}
