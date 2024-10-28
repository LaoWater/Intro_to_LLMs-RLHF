def find_palindromic_substrings(s):
    n = len(s)
    # Table to store whether a substring is a palindrome
    dp = [[False] * n for _ in range(n)]
    palindromes = []

    # Every single character is a palindrome
    for i in range(n):
        dp[i][i] = True
        # palindromes.append(s[i])

    # Check for substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            palindromes.append(s[i:i + 2])

    # Check for substrings of length greater than 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            # Substring s[i:j+1] is a palindrome if s[i] == s[j] and s[i+1:j] is a palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                palindromes.append(s[i:j + 1])

    return palindromes


# Example usage
input_string = "racecar"
print("Palindromic substrings:", find_palindromic_substrings(input_string))
