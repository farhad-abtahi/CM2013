classdef test_data_loader < matlab.unittest.TestCase
    methods (Test)
        function testLoadTrainingData(testCase)
            % Assuming dummy.edf exists in data/training for path reference
            dummyEdfPath = fullfile('..', 'data', 'training', 'dummy.edf');
            [eeg_data, labels] = data_loader_load_training_data(dummyEdfPath);
            
            testCase.verifyTrue(isnumeric(eeg_data));
            testCase.verifyTrue(isnumeric(labels));
            testCase.verifyEqual(size(eeg_data), [20, 3000]); % 20 epochs, 3000 samples per epoch
            testCase.verifyEqual(size(labels), [1, 20]);
        end
        
        function testLoadHoldoutData(testCase)
            % Assuming dummy_holdout.edf exists in data/holdout for path reference
            dummyHoldoutPath = fullfile('..', 'data', 'holdout', 'dummy_holdout.edf');
            eeg_data = data_loader_load_holdout_data(dummyHoldoutPath);
            
            testCase.verifyTrue(isnumeric(eeg_data));
            testCase.verifyEqual(size(eeg_data), [20, 3000]); % 20 epochs, 3000 samples per epoch
        end
    end
end
