    package com.secureai.utils;

    import com.secureai.utils.AttackType;

    public class RewardShapingUtils {
        public static double calculateReward(AttackType actual, AttackType predicted, int detectionTime) {
            if (actual == AttackType.NONE) return 1.0;
            if (actual == predicted) return 2.0 - 0.1 * detectionTime;
            else return -2.0;
        }

    }
