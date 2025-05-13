package com.secureai.utils;

import com.secureai.utils.AttackType;
import java.util.Random;

public class AttackScenarioGenerator {
    private static final Random random = new Random();

    public static AttackType generate() {
        AttackType[] types = AttackType.values();
        return types[random.nextInt(types.length)];
    }
}
