classdef test_preprocessing < matlab.unittest.TestCase
    methods (Test)
        function testLowpassFilter(testCase)
            fs = 100;
            cutoff = 10;
            t = 0:1/fs:10-1/fs;
            data = sin(2 * pi * 5 * t) + sin(2 * pi * 20 * t);
            filtered_data = lowpass_filter(data, cutoff, fs);
            
            testCase.verifyTrue(isnumeric(filtered_data));
            testCase.verifyEqual(size(filtered_data), size(data));
            % Basic check: ensure some attenuation of high frequency component
            % This is a very simple check and might need more sophisticated validation
            testCase.verifyLessThan(std(filtered_data), std(data));
        end
        
        function testPreprocessIteration1(testCase)
            % Dummy data: 20 epochs, 3000 samples per epoch
            eeg_data = randn(20, 3000);
            
            % Create a dummy config struct
            config.CURRENT_ITERATION = 1;
            config.LOW_PASS_FILTER_FREQ = 40;
            
            preprocessed_data = preprocessing_preprocess(eeg_data, config);
            
            testCase.verifyTrue(isnumeric(preprocessed_data));
            testCase.verifyEqual(size(preprocessed_data), size(eeg_data));
        end
        
        function testPreprocessOtherIteration(testCase)
            % Dummy data: 20 epochs, 3000 samples per epoch
            eeg_data = randn(20, 3000);
            
            % Create a dummy config struct for an unhandled iteration
            config.CURRENT_ITERATION = 99;
            config.LOW_PASS_FILTER_FREQ = 40;
            
            preprocessed_data = preprocessing_preprocess(eeg_data, config);
            
            testCase.verifyTrue(isnumeric(preprocessed_data));
            testCase.verifyEqual(size(preprocessed_data), size(eeg_data));
            % For other iterations, it should return raw data, so it should be very similar
            testCase.verifyEqual(preprocessed_data, eeg_data, 'AbsTol', 1e-9); % Check if it's essentially the same
        end
    end
end
