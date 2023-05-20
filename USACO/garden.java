import java.io.*;
import java.util.*;
import java.lang.*;
public class garden
{
   static char[][] a;
   static HashSet<String> set = new HashSet<>();
   static int count = 0;
   static int n;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("garden.in"));
      n = in.nextInt();
      a = new char[n][n];
      for(int i = 0; i < n; i++) a[i] = in.next().toCharArray();
      
      for(int i = 0; i < n; i++)
         solve(n - 1 - i, i, a[n - 1 - i][i] + "");
      
      for(int i = 0; i < n; i++)
         solve2(n - 1 - i, i, a[n - 1 - i][i] + "");
      
      System.out.println(count);
   }
   
   public static void solve(int i, int j, String s)
   {
      if(i == 0 && j == 0)
      {
         set.add(s);
         return;
      }
      
      if(i - 1 >= 0)
         solve(i - 1, j, s + a[i - 1][j]);
      if(j - 1 >= 0)
         solve(i, j - 1, s + a[i][j - 1]);    
   }
   
   public static void solve2(int i, int j, String s)
   {
      if(i == n - 1 && j == n - 1)
      {
         if(set.contains(s))
         {
            count++;
         }
         set.remove(s);
         return;
      }
      
      if(i + 1 < n)
         solve2(i + 1, j, s + a[i + 1][j]);
      if(j + 1 < n)
         solve2(i, j + 1, s + a[i][j + 1]);    
   }
}