import java.util.*;
import java.io.*;
import java.lang.*;

public class carrot
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("carrot.in"));
      int n = in.nextInt();
      int[] a = new int[n];
      int oneR = 0, twoL = 0;
      for(int i = 0; i < n; i++)
         a[i] = in.nextInt();
      
      in.close();
      
      int ans = 0;
      
      for(int i = 0; i < n; i++)
         if(a[i] == 1) oneR++;
      ans = oneR;
      
      for(int i = 0; i < n; i++)
      {
         if(a[i] == 1) oneR--;
         else twoL++;
         ans = Math.min(ans, oneR + twoL);
      }
      
      System.out.println(ans);
   }
}